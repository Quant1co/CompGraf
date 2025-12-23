#include "Game.hpp"

#include <random>
#include <chrono>
#include <iostream>

static constexpr float PI = 3.1415926535f;
static constexpr int TARGET_COUNT = 6;
static constexpr int CLOUD_COUNT = 6;
static constexpr int BALLOON_COUNT = 3;
static constexpr int LAMP_COUNT = 4;

Game::Game()
    : window(sf::VideoMode(1280, 720), "Airship Delivery", sf::Style::Default,
        [] {
            sf::ContextSettings settings;
            settings.depthBits = 24;
            settings.stencilBits = 8;
            settings.antialiasingLevel = 4;
            settings.majorVersion = 3;
            settings.minorVersion = 3;
            settings.attributeFlags = sf::ContextSettings::Core;
            return settings;
        }())
{
    window.setVerticalSyncEnabled(true);
}

bool Game::init()
{
    if (!gladLoadGL())
    {
        std::cerr << "Не удалось инициализировать GLAD" << std::endl;
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const char* litVertex = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aUV;
        uniform mat4 uMVP;
        uniform mat4 uModel;
        uniform mat3 uNormalMatrix;
        uniform float uWaveAmount;
        uniform float uTime;
        out vec3 vNormal;
        out vec3 vWorld;
        out vec2 vUV;
        void main()
        {
            vec3 pos = aPos;
            if(uWaveAmount > 0.0){
                pos.y += sin(uTime + pos.x*0.4 + pos.z*0.3) * uWaveAmount;
            }
            vec4 world = uModel * vec4(pos, 1.0);
            vWorld = world.xyz;
            vNormal = normalize(uNormalMatrix * aNormal);
            vUV = aUV;
            gl_Position = uMVP * vec4(pos, 1.0);
        }
    )";

    const char* litFragment = R"(
        #version 330 core
        struct PointLight { vec3 pos; vec3 color; float intensity; };
        uniform int uPointCount;
        uniform PointLight uPointLights[8];
        uniform vec3 uDirLightDir;
        uniform vec3 uDirLightColor;
        uniform vec3 uSpotPos;
        uniform vec3 uSpotDir;
        uniform vec3 uSpotColor;
        uniform float uSpotCutoff;
        uniform int uSpotEnabled;
        uniform vec3 uViewPos;
        uniform vec3 uColor;
        uniform float uEmission;
        uniform float uAlpha;
        in vec3 vNormal;
        in vec3 vWorld;
        out vec4 fragColor;
        void main()
        {
            vec3 N = normalize(vNormal);
            vec3 light = vec3(0.1);

            float ndl = max(dot(N, -normalize(uDirLightDir)), 0.0);
            light += uDirLightColor * ndl;

            for(int i=0;i<uPointCount;i++){
                vec3 L = uPointLights[i].pos - vWorld;
                float dist = length(L);
                L /= dist;
                float att = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);
                light += uPointLights[i].color * max(dot(N, L),0.0) * att * uPointLights[i].intensity;
            }

            if(uSpotEnabled==1){
                vec3 L = normalize(uSpotPos - vWorld);
                float theta = dot(L, -normalize(uSpotDir));
                if(theta > uSpotCutoff){
                    float falloff = (theta - uSpotCutoff) / (1.0 - uSpotCutoff);
                    light += uSpotColor * max(dot(N, L),0.0) * falloff;
                }
            }

            vec3 color = uColor * light + uEmission;
            fragColor = vec4(color, uAlpha);
        }
    )";

    const char* targetFragment = R"(
        #version 330 core
        uniform float uTime;
        in vec3 vNormal;
        in vec3 vWorld;
        in vec2 vUV;
        out vec4 fragColor;
        void main()
        {
            vec2 uv = vUV;
            uv = uv - vec2(0.5);
            float dist = length(uv);
            float rings = fract(dist * 10.0 - uTime * 1.5);
            float mask = smoothstep(0.1, 0.0, abs(rings - 0.5));
            vec3 base = vec3(0.85, 0.85, 0.85);
            vec3 red = vec3(0.8, 0.1, 0.1);
            vec3 color = mix(base, red, mask);
            float light = max(dot(normalize(vNormal), normalize(vec3(0.3,1.0,0.2))), 0.2);
            fragColor = vec4(color * light, 1.0);
        }
    )";

    litProgram = linkProgram(litVertex, litFragment);
    targetProgram = linkProgram(litVertex, targetFragment);

    cube = makeCube();
    quad = makeQuad();
    if (!loadHeightmap("heightmap.png", terrainGrid, 2.5f, terrainHeights))
    {
        terrainHeights.resize(terrainGrid * terrainGrid);
        for (int z = 0; z < terrainGrid; ++z)
        {
            for (int x = 0; x < terrainGrid; ++x)
            {
                float h = std::sin(x * 0.2f) * 0.3f + std::cos(z * 0.2f) * 0.3f;
                terrainHeights[z * terrainGrid + x] = h;
            }
        }
    }
    terrain = makeTerrain(terrainGrid, terrainScale, terrainHeights);

    setupScene();
    return true;
}

void Game::setupScene()
{
    airship.position = { 0.0f, 5.0f, 0.0f };
    airship.scale = { 2.0f, 1.0f, 1.0f };
    airship.color = { 0.7f, 0.7f, 0.9f };
    airship.radius = 2.0f;

    targets.assign(TARGET_COUNT, {});
    for (int i = 0; i < TARGET_COUNT; ++i)
    {
        targets[i].scale = { 2.5f, 0.1f, 2.5f };
        targets[i].color = { 1.0f, 0.3f, 0.3f };
        targets[i].radius = 2.0f;
    }
    for (int i = 0; i < TARGET_COUNT; ++i)
        respawnTarget(targets[i], targets, 0.02f);

    decorations.clear();
    for (int i = 0; i < 4; ++i)
    {
        Entity deco;
        deco.position = randomXZ(20.0f);
        deco.position.y = 0.2f;
        deco.scale = { 1.5f, 1.5f, 1.5f };
        deco.color = { 0.6f, 0.4f, 0.2f };
        deco.radius = 1.5f;
        decorations.push_back(deco);
    }

    clouds.resize(CLOUD_COUNT);
    for (int i = 0; i < CLOUD_COUNT; ++i)
    {
        clouds[i].basePosition = randomXZ(30.0f) + glm::vec3(0.0f, randFloat(6.0f, 10.0f), 0.0f);
        clouds[i].radius = randFloat(1.5f, 2.5f);
        clouds[i].phase = randFloat(0.0f, 6.28f);
    }

    balloons.resize(BALLOON_COUNT);
    for (auto& b : balloons)
    {
        b.position = randomXZ(15.0f) + glm::vec3(0.0f, randFloat(4.0f, 8.0f), 0.0f);
        b.scale = { 0.8f, 1.1f, 0.8f };
        b.color = { 0.9f, 0.7f, 0.2f };
        b.radius = 0.8f;
    }

    lamps.resize(LAMP_COUNT);
    for (int i = 0; i < LAMP_COUNT; ++i)
    {
        lamps[i].position = glm::vec3(std::cos(i * PI * 0.5f) * 6.0f, 0.1f, std::sin(i * PI * 0.5f) * 6.0f);
        lamps[i].scale = { 0.2f, 2.0f, 0.2f };
        lamps[i].color = { 1.0f, 0.8f, 0.4f };
        lamps[i].radius = 0.5f;
    }

    parcels.assign(32, {});
    gifts.clear();
    parcelIndex = 0;
    score = 0;
    spotlightOn = true;
    yaw = 0.0f;
    aimCamera = false;
    clock.restart();
}

void Game::run()
{
    while (window.isOpen())
    {
        processEvents();
        float dt = clock.restart().asSeconds();
        float time = clock.getElapsedTime().asSeconds();
        update(dt, time);
        render(time);
        window.display();
    }
}

void Game::processEvents()
{
    sf::Event ev{};
    while (window.pollEvent(ev))
    {
        if (ev.type == sf::Event::Closed)
            window.close();
        if (ev.type == sf::Event::KeyPressed)
        {
            if (ev.key.code == sf::Keyboard::Escape)
                window.close();
            if (ev.key.code == sf::Keyboard::L)
                spotlightOn = !spotlightOn;
            if (ev.key.code == sf::Keyboard::C)
                aimCamera = !aimCamera;
            if (ev.key.code == sf::Keyboard::Space)
            {
                parcels[parcelIndex].active = true;
                parcels[parcelIndex].position = airship.position + glm::vec3(0.0f, -0.5f, 0.0f);
                parcels[parcelIndex].velocity = { 0.0f, -1.0f, 0.0f };
                parcelIndex = (parcelIndex + 1) % static_cast<int>(parcels.size());
            }
        }
    }
}

void Game::update(float dt, float time)
{
    glm::vec3 move{ 0.0f };
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) move.z += 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) move.z -= 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) move.x -= 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) move.x += 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) move.y += 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::F)) move.y -= 1.0f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) yaw -= 1.5f * dt;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) yaw += 1.5f * dt;

    glm::mat4 rot = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 dir = glm::normalize(glm::vec3(rot * glm::vec4(move, 0.0f)) + glm::vec3(0.0f, move.y, 0.0f));
    if (glm::length(move) > 0.0f)
    {
        airship.position += dir * dt * 6.0f;
    }

    airship.position.y = std::clamp(airship.position.y, 2.0f, 20.0f);

    for (auto& p : parcels)
    {
        if (!p.active) continue;
        p.velocity.y -= 9.8f * dt;
        p.position += p.velocity * dt;
        float ground = sampleHeight(terrainHeights, terrainGrid, p.position.x, p.position.z);
        if (p.position.y < ground + 0.2f)
            p.active = false;

        for (auto& t : targets)
        {
            if (!t.visible) continue;
            if (glm::length(t.position - p.position) < (t.radius + p.radius))
            {
                t.visible = false;
                p.active = false;
                ++score;
                gifts.push_back(treePos + glm::vec3(randFloat(-1.5f, 1.5f), 0.3f, randFloat(-1.5f, 1.5f)));
            }
        }
    }

    for (auto& t : targets)
    {
        if (!t.visible)
        {
            respawnTarget(t, targets, 0.02f);
            t.visible = true;
        }
    }

    for (auto& c : clouds)
    {
        float phase = c.phase + time * 0.3f;
        c.basePosition.x += std::sin(phase) * 0.01f;
        c.basePosition.z += std::cos(phase) * 0.01f;
    }
}

void Game::render(float time)
{
    glViewport(0, 0, window.getSize().x, window.getSize().y);
    glClearColor(0.1f, 0.15f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Camera cam;
    cam.projection = glm::perspective(glm::radians(60.0f), window.getSize().x / static_cast<float>(window.getSize().y), 0.1f, 200.0f);

    glm::mat4 rot = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    if (aimCamera)
    {
        cam.position = airship.position + glm::vec3(0.0f, -0.5f, 0.0f);
        cam.target = cam.position + glm::vec3(0.0f, -1.0f, 0.0f);
    }
    else
    {
        glm::vec3 back = glm::vec3(rot * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
        cam.position = airship.position - back * 8.0f + glm::vec3(0.0f, 4.0f, 0.0f);
        cam.target = airship.position;
    }
    cam.view = glm::lookAt(cam.position, cam.target, glm::vec3(0.0f, 1.0f, 0.0f));

    auto setCommonUniforms = [&](GLuint prog, float emission, float alpha)
    {
        glUseProgram(prog);
        glUniform3f(glGetUniformLocation(prog, "uDirLightDir"), -0.3f, -1.0f, -0.2f);
        glUniform3f(glGetUniformLocation(prog, "uDirLightColor"), 0.9f, 0.9f, 0.8f);
        glUniform3f(glGetUniformLocation(prog, "uViewPos"), cam.position.x, cam.position.y, cam.position.z);
        glUniform1f(glGetUniformLocation(prog, "uEmission"), emission);
        glUniform1f(glGetUniformLocation(prog, "uAlpha"), alpha);

        glUniform1i(glGetUniformLocation(prog, "uPointCount"), LAMP_COUNT);
        for (int i = 0; i < LAMP_COUNT; ++i)
        {
            std::string base = "uPointLights[" + std::to_string(i) + "]";
            glUniform3f(glGetUniformLocation(prog, (base + ".pos").c_str()), lamps[i].position.x, lamps[i].position.y + 1.5f, lamps[i].position.z);
            glUniform3f(glGetUniformLocation(prog, (base + ".color").c_str()), 1.0f, 0.8f, 0.6f);
            glUniform1f(glGetUniformLocation(prog, (base + ".intensity").c_str()), 1.0f);
        }

        glUniform3f(glGetUniformLocation(prog, "uSpotPos"), airship.position.x, airship.position.y - 0.5f, airship.position.z);
        glm::vec3 spotDir = glm::vec3(rot * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
        glUniform3f(glGetUniformLocation(prog, "uSpotDir"), spotDir.x, spotDir.y, spotDir.z);
        glUniform3f(glGetUniformLocation(prog, "uSpotColor"), 1.0f, 1.0f, 0.8f);
        glUniform1f(glGetUniformLocation(prog, "uSpotCutoff"), std::cos(glm::radians(20.0f)));
        glUniform1i(glGetUniformLocation(prog, "uSpotEnabled"), spotlightOn ? 1 : 0);
        glUniform1f(glGetUniformLocation(prog, "uTime"), time);
    };

    auto drawMesh = [&](const Mesh& mesh, const Entity& e, GLuint prog, float wave, float emission, float alpha)
    {
        glm::mat4 model = composeModel(e);
        glm::mat4 mvp = cam.projection * cam.view * model;
        glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(model)));
        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
        glUniform3f(glGetUniformLocation(prog, "uColor"), e.color.x, e.color.y, e.color.z);
        glUniform1f(glGetUniformLocation(prog, "uWaveAmount"), wave);
        setCommonUniforms(prog, emission, alpha);
        glBindVertexArray(mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
        glBindVertexArray(0);
    };

    Entity ground;
    ground.position = { 0.0f, 0.0f, 0.0f };
    ground.scale = { 1.0f, 1.0f, 1.0f };
    ground.color = { 0.2f, 0.4f, 0.2f };
    drawMesh(terrain, ground, litProgram, 0.05f, 0.0f, 1.0f);

    Entity trunk;
    trunk.position = treePos + glm::vec3(0.0f, 1.0f, 0.0f);
    trunk.scale = { 0.6f, 2.0f, 0.6f };
    trunk.color = { 0.4f, 0.25f, 0.1f };
    drawMesh(cube, trunk, litProgram, 0.0f, 0.0f, 1.0f);
    Entity crown;
    crown.position = treePos + glm::vec3(0.0f, 3.0f, 0.0f);
    crown.scale = { 2.5f, 2.5f, 2.5f };
    crown.color = { 0.1f, 0.5f, 0.2f };
    drawMesh(cube, crown, litProgram, 0.08f, 0.0f, 1.0f);

    for (auto& l : lamps)
        drawMesh(cube, l, litProgram, 0.0f, 0.0f, 1.0f);

    Entity sleigh;
    sleigh.scale = { 1.2f, 0.4f, 2.0f };
    sleigh.color = { 0.7f, 0.1f, 0.1f };
    float angle = time * 0.5f;
    sleigh.position = treePos + glm::vec3(std::cos(angle) * 5.0f, 0.5f, std::sin(angle) * 5.0f);
    drawMesh(cube, sleigh, litProgram, 0.0f, 0.0f, 1.0f);

    for (auto& t : targets)
    {
        if (!t.visible) continue;
        drawMesh(quad, t, targetProgram, 0.0f, 0.0f, 1.0f);
    }

    for (auto& d : decorations)
        drawMesh(cube, d, litProgram, 0.03f, 0.0f, 1.0f);

    drawMesh(cube, airship, litProgram, 0.05f, 0.0f, 1.0f);

    Entity parcelEntity;
    parcelEntity.scale = { 0.4f, 0.4f, 0.4f };
    parcelEntity.color = { 0.8f, 0.6f, 0.3f };
    for (auto& p : parcels)
    {
        if (!p.active) continue;
        parcelEntity.position = p.position;
        drawMesh(cube, parcelEntity, litProgram, 0.0f, 0.0f, 1.0f);
    }

    Entity gift;
    gift.scale = { 0.5f, 0.5f, 0.5f };
    gift.color = { 0.2f, 0.2f, 0.8f };
    for (auto& g : gifts)
    {
        gift.position = g;
        drawMesh(cube, gift, litProgram, 0.0f, 0.0f, 1.0f);
    }

    Entity cloudEntity;
    cloudEntity.scale = { 3.0f, 1.5f, 2.0f };
    cloudEntity.color = { 0.9f, 0.9f, 0.95f };
    bool flash = std::fmod(time, 6.0f) < 0.5f;
    for (auto& c : clouds)
    {
        cloudEntity.position = c.basePosition;
        cloudEntity.scale = glm::vec3(c.radius * 1.3f, c.radius * 0.8f, c.radius * 1.2f);
        float emission = flash ? 1.2f : 0.0f;
        drawMesh(cube, cloudEntity, litProgram, 0.1f, emission, 0.6f);
    }

    for (auto& b : balloons)
        drawMesh(cube, b, litProgram, 0.05f, 0.0f, 0.9f);
}

GLuint Game::compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status)
    {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(length), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        std::cerr << "Shader compile error: " << log << "\n";
    }
    return shader;
}

GLuint Game::linkProgram(const char* vs, const char* fs)
{
    GLuint program = glCreateProgram();
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(program, v);
    glAttachShader(program, f);
    glLinkProgram(program);
    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status)
    {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(length), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        std::cerr << "Link error: " << log << "\n";
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return program;
}

Mesh Game::makeCube()
{
    const float verts[] = {
        -0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,0.0f,
         0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,0.0f,
         0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,1.0f,
        -0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,0.0f,
         0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,1.0f,
        -0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,1.0f,
        -0.5f,-0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,0.0f,
         0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 1.0f,1.0f,
         0.5f,-0.5f,-0.5f, 0.0f,0.0f,-1.0f, 1.0f,0.0f,
        -0.5f,-0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,0.0f,
        -0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,1.0f,
         0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 1.0f,1.0f,
        -0.5f, 0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,1.0f,
        -0.5f, 0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,1.0f,
        -0.5f,-0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,0.0f,
        -0.5f, 0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,1.0f,
        -0.5f,-0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,0.0f,
        -0.5f,-0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,0.0f,
         0.5f, 0.5f, 0.5f, 1.0f,0.0f,0.0f, 1.0f,1.0f,
         0.5f,-0.5f,-0.5f, 1.0f,0.0f,0.0f, 0.0f,0.0f,
         0.5f, 0.5f,-0.5f, 1.0f,0.0f,0.0f, 0.0f,1.0f,
         0.5f, 0.5f, 0.5f, 1.0f,0.0f,0.0f, 1.0f,1.0f,
         0.5f,-0.5f, 0.5f, 1.0f,0.0f,0.0f, 1.0f,0.0f,
         0.5f,-0.5f,-0.5f, 1.0f,0.0f,0.0f, 0.0f,0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f,1.0f,0.0f, 0.0f,1.0f,
         0.5f, 0.5f,-0.5f, 0.0f,1.0f,0.0f, 1.0f,0.0f,
         0.5f, 0.5f, 0.5f, 0.0f,1.0f,0.0f, 1.0f,1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f,1.0f,0.0f, 0.0f,1.0f,
        -0.5f, 0.5f,-0.5f, 0.0f,1.0f,0.0f, 0.0f,0.0f,
         0.5f, 0.5f,-0.5f, 0.0f,1.0f,0.0f, 1.0f,0.0f,
        -0.5f,-0.5f, 0.5f, 0.0f,-1.0f,0.0f, 0.0f,1.0f,
         0.5f,-0.5f, 0.5f, 0.0f,-1.0f,0.0f, 1.0f,1.0f,
         0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f, 1.0f,0.0f,
        -0.5f,-0.5f, 0.5f, 0.0f,-1.0f,0.0f, 0.0f,1.0f,
         0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f, 1.0f,0.0f,
        -0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f, 0.0f,0.0f
    };

    Mesh m{};
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 6));
    glBindVertexArray(0);
    m.vertexCount = 36;
    return m;
}

Mesh Game::makeQuad()
{
    const float verts[] = {
        -0.5f, 0.0f, -0.5f, 0.0f,1.0f,0.0f, 0.0f,0.0f,
         0.5f, 0.0f, -0.5f, 0.0f,1.0f,0.0f, 1.0f,0.0f,
         0.5f, 0.0f,  0.5f, 0.0f,1.0f,0.0f, 1.0f,1.0f,
        -0.5f, 0.0f, -0.5f, 0.0f,1.0f,0.0f, 0.0f,0.0f,
         0.5f, 0.0f,  0.5f, 0.0f,1.0f,0.0f, 1.0f,1.0f,
        -0.5f, 0.0f,  0.5f, 0.0f,1.0f,0.0f, 0.0f,1.0f,
    };

    Mesh m{};
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 6));
    glBindVertexArray(0);
    m.vertexCount = 6;
    return m;
}

Mesh Game::makeTerrain(int grid, float scale, const std::vector<float>& heights)
{
    std::vector<float> verts;
    verts.reserve((grid - 1) * (grid - 1) * 6 * 8);
    auto idx = [grid](int x, int z) { return z * grid + x; };
    auto heightAt = [&](int x, int z) { return heights[idx(std::clamp(x, 0, grid - 1), std::clamp(z, 0, grid - 1))]; };

    for (int z = 0; z < grid - 1; ++z)
    {
        for (int x = 0; x < grid - 1; ++x)
        {
            glm::vec3 p0 = { (x - grid / 2) * scale, heightAt(x, z), (z - grid / 2) * scale };
            glm::vec3 p1 = { (x + 1 - grid / 2) * scale, heightAt(x + 1, z), (z - grid / 2) * scale };
            glm::vec3 p2 = { (x + 1 - grid / 2) * scale, heightAt(x + 1, z + 1), (z + 1 - grid / 2) * scale };
            glm::vec3 p3 = { (x - grid / 2) * scale, heightAt(x, z + 1), (z + 1 - grid / 2) * scale };

            glm::vec3 n0 = glm::normalize(glm::cross(p1 - p0, p3 - p0));
            glm::vec3 n1 = glm::normalize(glm::cross(p2 - p1, p3 - p1));

            auto pushVert = [&](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv)
            {
                verts.insert(verts.end(), { p.x, p.y, p.z, n.x, n.y, n.z, uv.x, uv.y });
            };

            pushVert(p0, n0, { 0.0f, 0.0f });
            pushVert(p1, n0, { 1.0f, 0.0f });
            pushVert(p3, n0, { 0.0f, 1.0f });

            pushVert(p1, n1, { 1.0f, 0.0f });
            pushVert(p2, n1, { 1.0f, 1.0f });
            pushVert(p3, n1, { 0.0f, 1.0f });
        }
    }

    Mesh m{};
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 6));
    glBindVertexArray(0);
    m.vertexCount = static_cast<GLsizei>(verts.size() / 8);
    return m;
}

float Game::sampleHeight(const std::vector<float>& h, int grid, float x, float z)
{
    // Context for later grid use
    float fx = (x / 1.0f + grid / 2.0f);
    float fz = (z / 1.0f + grid / 2.0f);
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    ix = std::clamp(ix, 0, grid - 2);
    iz = std::clamp(iz, 0, grid - 2);
    float tx = fx - ix;
    float tz = fz - iz;
    auto idx = [grid](int x, int z) { return z * grid + x; };
    float h00 = h[idx(ix, iz)];
    float h10 = h[idx(ix + 1, iz)];
    float h01 = h[idx(ix, iz + 1)];
    float h11 = h[idx(ix + 1, iz + 1)];
    float hx0 = std::lerp(h00, h10, tx);
    float hx1 = std::lerp(h01, h11, tx);
    return std::lerp(hx0, hx1, tz);
}

bool Game::loadHeightmap(const std::string& path, int grid, float amplitude, std::vector<float>& heightsOut)
{
    sf::Image img;
    if (!img.loadFromFile(path))
        return false;

    heightsOut.resize(grid * grid);
    auto sample = [&](float u, float v) {
        unsigned x = static_cast<unsigned>(u * (img.getSize().x - 1));
        unsigned y = static_cast<unsigned>(v * (img.getSize().y - 1));
        sf::Color c = img.getPixel(x, y);
        float g = static_cast<float>(c.r) / 255.0f; // assume grayscale
        return (g - 0.5f) * amplitude;
    };

    for (int z = 0; z < grid; ++z)
    {
        for (int x = 0; x < grid; ++x)
        {
            float u = x / static_cast<float>(grid - 1);
            float v = z / static_cast<float>(grid - 1);
            heightsOut[z * grid + x] = sample(u, v);
        }
    }
    return true;
}

glm::mat4 Game::composeModel(const Entity& e)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, e.position);
    model = glm::scale(model, e.scale);
    return model;
}

float Game::randFloat(float min, float max)
{
    static std::mt19937 rng(static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

glm::vec3 Game::randomXZ(float range)
{
    return { randFloat(-range, range), 0.0f, randFloat(-range, range) };
}

bool Game::overlaps(const Entity& a, const Entity& b)
{
    return glm::length(a.position - b.position) < (a.radius + b.radius);
}

void Game::respawnTarget(Entity& target, const std::vector<Entity>& all, float groundOffset)
{
    bool good = false;
    int tries = 0;
    while (!good && tries < 64)
    {
        ++tries;
        target.position = randomXZ(25.0f);
        target.position.y = groundOffset;
        good = true;
        for (const auto& other : all)
        {
            if (&other != &target && overlaps(target, other))
            {
                good = false;
                break;
            }
        }
    }
}
