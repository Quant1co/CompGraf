#include "Game.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <random>
#include <chrono>
#include <iostream>
#include <algorithm> // Важно для std::clamp
#include <cmath>     // Важно для sin/cos
#include <numbers>   // Для PI

static constexpr float PI = 3.1415926535f;
static constexpr int TARGET_COUNT = 6;
static constexpr int CLOUD_COUNT = 6;
static constexpr int BALLOON_COUNT = 3;
static constexpr int LAMP_COUNT = 4;

static GLuint unlitProgram = 0;
static GLuint unlitWaveProgram = 0;

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

Game::~Game()
{
	cleanup();
}

void Game::cleanup()
{
	if (litProgram) glDeleteProgram(litProgram);
	if (targetProgram) glDeleteProgram(targetProgram);
	if (unlitProgram) glDeleteProgram(unlitProgram);
	if (unlitWaveProgram) glDeleteProgram(unlitWaveProgram);

	auto deleteMesh = [](Mesh& m) {
		if (m.vbo) glDeleteBuffers(1, &m.vbo);
		if (m.vao) glDeleteVertexArrays(1, &m.vao);
		};

	deleteMesh(cube);
	deleteMesh(quad);
	deleteMesh(sphere);
	deleteMesh(cone);
	deleteMesh(terrain);
	deleteMesh(airshipModel);
	deleteMesh(treeModel);
}

bool Game::init()
{
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(sf::Context::getFunction)))
	{
		std::cerr << "Не удалось инициализировать GLAD" << std::endl;
		return false;
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

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

	const char* unlitVertex = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )";

	const char* unlitWaveVertex = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        uniform float uWaveAmount;
        uniform float uTime;
        void main()
        {
            vec3 pos = aPos;
            if(uWaveAmount > 0.0){
                // Колыхание усиливается к вершине (чем выше - тем сильнее)
                float heightFactor = (pos.y + 0.5) * 0.5;
                pos.x += sin(uTime * 2.0 + pos.y * 3.0) * uWaveAmount * heightFactor;
                pos.z += cos(uTime * 1.5 + pos.y * 2.0) * uWaveAmount * heightFactor * 0.5;
            }
            gl_Position = uMVP * vec4(pos, 1.0);
        }
    )";

	const char* unlitFragment = R"(
		#version 330 core
		uniform vec3 uColor;
		uniform float uBrightness; // Новый параметр
		out vec4 fragColor;
		void main()
		{
			fragColor = vec4(uColor + vec3(uBrightness), 1.0);
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
            vec3 light = vec3(0.1); // Ambient

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
	unlitProgram = linkProgram(unlitVertex, unlitFragment);
	unlitWaveProgram = linkProgram(unlitWaveVertex, unlitFragment);

	cube = makeCube();
	quad = makeQuad();
	sphere = makeSphere(16, 32);
	cone = makeCone(16); // Конус для ёлки

	// Загрузка OBJ моделей (расширение .model чтобы линкер не путал)
	airshipModel = loadOBJ("airship.model");
	treeModel = loadOBJ("tree.model");

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
	// Форма сигары вытянутая по Z (направление движения)
	airship.scale = { 0.8f, 0.8f, 2.5f };
	airship.color = { 0.8f, 0.8f, 0.9f }; // Белый/серебристый
	airship.radius = 2.0f;
	airshipVelocity = { 0.0f, 0.0f, 0.0f };

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

	// Второй тип декораций (например, маленькие ёлочки/кусты)
	for (int i = 0; i < 4; ++i)
	{
		Entity deco;
		deco.position = randomXZ(20.0f);
		deco.position.y = 0.2f;
		deco.scale = { 0.8f, 1.2f, 0.8f };
		deco.color = { 0.15f, 0.45f, 0.2f };
		deco.radius = 1.0f;
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

	// Сани вокруг ёлки
	sleigh.scale = { 1.2f, 0.4f, 0.6f };
	sleigh.color = { 0.8f, 0.1f, 0.1f };
	sleigh.radius = 0.8f;
	sleighAngle = 0.0f;

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
		if (dt > 0.1f) dt = 0.1f;

		static float globalTime = 0.0f;
		globalTime += dt;

		update(dt, globalTime);
		render(globalTime);
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
				parcels[parcelIndex].velocity = airshipVelocity + glm::vec3(0.0f, -2.0f, 0.0f);
				parcelIndex = (parcelIndex + 1) % static_cast<int>(parcels.size());
			}
		}
	}
}

void Game::update(float dt, float time)
{
	glm::vec3 input{ 0.0f };
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) input.z += 1.0f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) input.z -= 1.0f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) input.x += 1.0f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) input.x -= 1.0f;

	float verticalInput = 0.0f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) verticalInput += 1.0f;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) verticalInput -= 1.0f;

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) yaw += 2.0f * dt;
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) yaw -= 2.0f * dt;

	glm::mat4 rot = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));

	float acceleration = 15.0f;
	float friction = 2.0f;

	if (glm::length(input) > 0.0f)
	{
		glm::vec3 dir = glm::normalize(glm::vec3(rot * glm::vec4(input, 0.0f)));
		airshipVelocity.x += dir.x * acceleration * dt;
		airshipVelocity.z += dir.z * acceleration * dt;
	}

	airshipVelocity.y += verticalInput * acceleration * dt;
	airshipVelocity -= airshipVelocity * friction * dt;

	airship.position += airshipVelocity * dt;

	if (airship.position.y < 2.0f) {
		airship.position.y = 2.0f;
		airshipVelocity.y = 0.0f;
	}
	if (airship.position.y > 20.0f) {
		airship.position.y = 20.0f;
		airshipVelocity.y = 0.0f;
	}

	for (auto& p : parcels)
	{
		if (!p.active) continue;
		p.velocity.y -= 9.8f * dt;
		p.velocity.x -= p.velocity.x * 0.5f * dt;
		p.velocity.z -= p.velocity.z * 0.5f * dt;

		p.position += p.velocity * dt;

		float ground = sampleHeight(terrainHeights, terrainGrid, p.position.x, p.position.z);
		if (p.position.y < ground + 0.2f)
			p.active = false;

		for (auto& t : targets)
		{
			if (!t.visible) continue;
			float hDiff = std::abs(p.position.y - t.position.y);
			float xzDist = glm::length(glm::vec2(t.position.x - p.position.x, t.position.z - p.position.z));

			if (xzDist < (t.radius + p.radius) && hDiff < 1.0f)
			{
				t.visible = false;
				p.active = false;
				++score;
				std::cout << "Score: " << score << std::endl;
				gifts.push_back(treePos + glm::vec3(randFloat(-1.5f, 1.5f), 0.3f, randFloat(-1.5f, 1.5f)));
			}
		}
	}

	for (auto& t : targets)
	{
		if (!t.visible)
		{
			respawnTarget(t, targets, 0.05f);
			t.visible = true;
		}
	}

	for (auto& c : clouds)
	{
		float phase = c.phase + time * 0.3f;
		c.basePosition.x += std::sin(phase) * 0.01f;
		c.basePosition.z += std::cos(phase) * 0.01f;
	}

	// Сани ездят вокруг ёлки
	sleighAngle += sleighSpeed * dt;
	float sx = std::cos(sleighAngle) * sleighRadius;
	float sz = std::sin(sleighAngle) * sleighRadius;
	sleigh.position = treePos + glm::vec3(sx, 0.25f, sz);
}

void Game::render(float time)
{
	glViewport(0, 0, window.getSize().x, window.getSize().y);
	glClearColor(0.1f, 0.15f, 0.25f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Рисуем непрозрачную геометрию без blending.
	// Blending включаем только для реально полупрозрачных объектов.
	glDisable(GL_BLEND);

	Camera cam;
	cam.projection = glm::perspective(glm::radians(60.0f), window.getSize().x / static_cast<float>(window.getSize().y), 0.1f, 200.0f);

	glm::mat4 rot = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));

	if (aimCamera)
	{
		// РЕЖИМ ПРИЦЕЛИВАНИЯ:
		// Опускаем камеру значительно ниже (на -1.5 или -2.0), чтобы она была под гондолой
		cam.position = airship.position + glm::vec3(0.0f, -2.0f, 0.0f);
		cam.target = cam.position + glm::vec3(0.0f, -10.0f, 0.0f); // Смотрим строго вниз

		// Вектор "вверх" для камеры совпадает с направлением движения дирижабля
		glm::vec3 forward = glm::vec3(rot * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
		cam.view = glm::lookAt(cam.position, cam.target, forward);
	}
	else
	{
		// ОБЫЧНЫЙ РЕЖИМ (Вид сзади):
		// Увеличиваем Z-отступ до -12.0 и высоту до 6.0. 
		// X ставим 0.0, чтобы камера была строго по центру за дирижаблем.
		glm::vec3 offset = glm::vec3(rot * glm::vec4(0.0f, 6.0f, -12.0f, 0.0f));

		cam.position = airship.position + offset;
		// Направляем взгляд чуть выше центра дирижабля для лучшего обзора горизонта
		cam.target = airship.position + glm::vec3(0.0f, 1.0f, 0.0f);
		cam.view = glm::lookAt(cam.position, cam.target, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	auto setCommonUniforms = [&](GLuint prog, float emission, float alpha)
		{
			glUseProgram(prog);
			glUniform3f(glGetUniformLocation(prog, "uDirLightDir"), -0.3f, -1.0f, -0.2f);
			glUniform3f(glGetUniformLocation(prog, "uDirLightColor"), 0.9f, 0.9f, 0.8f);
			glUniform3f(glGetUniformLocation(prog, "uViewPos"), cam.position.x, cam.position.y, cam.position.z);
			glUniform1f(glGetUniformLocation(prog, "uEmission"), emission);
			glUniform1f(glGetUniformLocation(prog, "uAlpha"), alpha);

			glUniform1i(glGetUniformLocation(prog, "uPointCount"), LAMP_COUNT + 1);
			for (int i = 0; i < LAMP_COUNT; ++i)
			{
				std::string treeLight = "uPointLights[" + std::to_string(LAMP_COUNT) + "]";
				glUniform3f(glGetUniformLocation(prog, (treeLight + ".pos").c_str()), treePos.x, treePos.y + 2.0f, treePos.z);
				glUniform3f(glGetUniformLocation(prog, (treeLight + ".color").c_str()), 0.2f, 0.8f, 0.2f); // Зеленоватый свет
				glUniform1f(glGetUniformLocation(prog, (treeLight + ".intensity").c_str()), 1.5f); // Интенсивность
			}

			glUniform3f(glGetUniformLocation(prog, "uSpotPos"), airship.position.x, airship.position.y - 0.5f, airship.position.z);
			glm::vec3 spotDir = glm::vec3(rot * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
			glUniform3f(glGetUniformLocation(prog, "uSpotDir"), spotDir.x, spotDir.y, spotDir.z);
			glUniform3f(glGetUniformLocation(prog, "uSpotColor"), 1.0f, 1.0f, 0.8f);
			glUniform1f(glGetUniformLocation(prog, "uSpotCutoff"), std::cos(glm::radians(20.0f)));
			glUniform1i(glGetUniformLocation(prog, "uSpotEnabled"), spotlightOn ? 1 : 0);
			glUniform1f(glGetUniformLocation(prog, "uTime"), time);
		};

	auto drawMeshUnlit = [&](const Mesh& mesh, const Entity& e, const glm::vec3& color)
		{
			glm::mat4 model = composeModel(e);
			glm::mat4 mvp = cam.projection * cam.view * model;
			glUseProgram(unlitProgram);
			glUniformMatrix4fv(glGetUniformLocation(unlitProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniform3f(glGetUniformLocation(unlitProgram, "uColor"), color.x, color.y, color.z);
			glBindVertexArray(mesh.vao);
			glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
			glBindVertexArray(0);
		};

	// Функция для рисования объектов без освещения, но с колыханием (для ёлочек)
	auto drawMeshUnlitWave = [&](const Mesh& mesh, const Entity& e, const glm::vec3& color, float wave, float brightness = 0.0f)
		{
			glm::mat4 model = composeModel(e);
			glm::mat4 mvp = cam.projection * cam.view * model;
			glUseProgram(unlitWaveProgram);
			glUniformMatrix4fv(glGetUniformLocation(unlitWaveProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniform3f(glGetUniformLocation(unlitWaveProgram, "uColor"), color.x, color.y, color.z);
			glUniform1f(glGetUniformLocation(unlitWaveProgram, "uWaveAmount"), wave);
			glUniform1f(glGetUniformLocation(unlitWaveProgram, "uTime"), time);
			// Передаем яркость
			glUniform1f(glGetUniformLocation(unlitWaveProgram, "uBrightness"), brightness);

			glBindVertexArray(mesh.vao);
			glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
			glBindVertexArray(0);
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
	// Зелёная трава
	ground.color = { 0.15f, 0.35f, 0.1f };
	drawMesh(terrain, ground, litProgram, 0.0f, 0.0f, 1.0f);

	// Ёлка: рисуем примитивами с лёгким колыханием
	{
		// Ствол (не колышется)
		Entity trunk;
		trunk.position = treePos + glm::vec3(0.0f, 0.5f, 0.0f);
		trunk.scale = { 0.4f, 1.0f, 0.4f };
		trunk.color = { 0.35f, 0.2f, 0.1f };
		drawMeshUnlit(cube, trunk, trunk.color);

		// Нижний ярус (лёгкое колыхание)
		Entity cone1;
		cone1.position = treePos + glm::vec3(0.0f, 2.0f, 0.0f);
		cone1.scale = { 1.8f, 1.2f, 1.8f };
		cone1.color = { 0.1f, 0.4f, 0.15f };
		drawMeshUnlitWave(cone, cone1, cone1.color, 0.03f);

		// Средний ярус (чуть сильнее колыхание)
		Entity cone2;
		cone2.position = treePos + glm::vec3(0.0f, 3.0f, 0.0f);
		cone2.scale = { 1.3f, 1.0f, 1.3f };
		cone2.color = { 0.1f, 0.45f, 0.15f };
		drawMeshUnlitWave(cone, cone2, cone2.color, 0.05f);

		// Верхний ярус (самое сильное колыхание)
		Entity cone3;
		cone3.position = treePos + glm::vec3(0.0f, 3.8f, 0.0f);
		cone3.scale = { 0.8f, 0.8f, 0.8f };
		cone3.color = { 0.1f, 0.5f, 0.15f };
		drawMeshUnlitWave(cone, cone3, cone3.color, 0.08f, 0.15f); 
	}

	// Звезда на верхушке ёлки (маленькая жёлтая сфера со свечением)
	Entity star;
	star.position = treePos + glm::vec3(0.0f, 4.5f, 0.0f);
	star.scale = { 0.4f, 0.4f, 0.4f };
	star.color = { 1.0f, 0.9f, 0.2f };
	drawMesh(sphere, star, litProgram, 0.0f, 0.8f, 1.0f);

	// Ёлочные шары (разноцветные) - размещены ближе к конусам ёлки
	const glm::vec3 ballColors[] = {
		{0.9f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.9f}, {0.9f, 0.8f, 0.1f},
		{0.1f, 0.8f, 0.9f}, {0.9f, 0.1f, 0.9f}, {0.1f, 0.9f, 0.2f}
	};
	for (int i = 0; i < 6; ++i)
	{
		float angle = i * PI / 3.0f;
		// Высота шаров на разных ярусах ёлки
		float height = 1.5f + (i % 3) * 0.7f;
		// Радиус уменьшается с высотой (соответствует форме конусов)
		float radius = 0.7f - (i % 3) * 0.15f;
		Entity ball;
		ball.position = treePos + glm::vec3(std::cos(angle) * radius, height, std::sin(angle) * radius);
		ball.scale = { 0.2f, 0.2f, 0.2f };
		ball.color = ballColors[i];
		drawMesh(sphere, ball, litProgram, 0.0f, 0.3f, 1.0f);
	}

	// Сани (едут вокруг ёлки)
	{
		Entity s = sleigh;
		// Поворачиваем сани по касательной траектории
		float heading = sleighAngle + PI * 0.5f;
		glm::mat4 model(1.0f);
		model = glm::translate(model, s.position);
		model = glm::rotate(model, heading, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::scale(model, s.scale);

		glm::mat4 mvp = cam.projection * cam.view * model;
		glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(model)));
		glUseProgram(litProgram);
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
		glUniform3f(glGetUniformLocation(litProgram, "uColor"), s.color.x, s.color.y, s.color.z);
		glUniform1f(glGetUniformLocation(litProgram, "uWaveAmount"), 0.0f);
		setCommonUniforms(litProgram, 0.0f, 1.0f);
		glBindVertexArray(cube.vao);
		glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
		glBindVertexArray(0);

		// Полозья саней (два куба снизу)
		for (int side = -1; side <= 1; side += 2)
		{
			glm::mat4 runnerModel(1.0f);
			runnerModel = glm::translate(runnerModel, s.position);
			runnerModel = glm::rotate(runnerModel, heading, glm::vec3(0.0f, 1.0f, 0.0f));
			runnerModel = glm::translate(runnerModel, glm::vec3(side * 0.4f, -0.15f, 0.0f));
			runnerModel = glm::scale(runnerModel, glm::vec3(0.1f, 0.1f, 0.8f));

			mvp = cam.projection * cam.view * runnerModel;
			normal = glm::mat3(glm::transpose(glm::inverse(runnerModel)));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(runnerModel));
			glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
			glUniform3f(glGetUniformLocation(litProgram, "uColor"), 0.3f, 0.2f, 0.1f);
			glBindVertexArray(cube.vao);
			glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
		}
		glBindVertexArray(0);
	}

	for (auto& l : lamps)
	{
		// Столб фонаря
		drawMesh(cube, l, litProgram, 0.0f, 0.0f, 1.0f);
		// Светящийся шар наверху
		Entity lampBulb;
		lampBulb.position = l.position + glm::vec3(0.0f, 2.2f, 0.0f);
		lampBulb.scale = { 0.4f, 0.4f, 0.4f };
		lampBulb.color = { 1.0f, 0.9f, 0.6f };
		drawMesh(sphere, lampBulb, litProgram, 0.0f, 0.8f, 1.0f);
	}

	for (auto& t : targets)
	{
		if (!t.visible) continue;
		// Мишень — площадка с красными кругами (оригинальный target)
		drawMesh(quad, t, targetProgram, 0.0f, 0.0f, 1.0f);
	}

	// Тень под дирижаблем (плоская проекция с учётом поворота)
	{
		glEnable(GL_BLEND);
		float groundH = sampleHeight(terrainHeights, terrainGrid, airship.position.x, airship.position.z);

		// Матрица тени с учётом поворота дирижабля
		glm::mat4 shadowModel(1.0f);
		shadowModel = glm::translate(shadowModel, glm::vec3(airship.position.x, groundH + 0.05f, airship.position.z));
		shadowModel = glm::rotate(shadowModel, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
		// Плоская тень: сплющиваем по Y, растягиваем по X/Z соответственно форме дирижабля
		shadowModel = glm::scale(shadowModel, glm::vec3(1.2f, 0.01f, 3.0f));

		glm::mat4 mvp = cam.projection * cam.view * shadowModel;
		glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(shadowModel)));

		glUseProgram(litProgram);
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(shadowModel));
		glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
		glUniform3f(glGetUniformLocation(litProgram, "uColor"), 0.0f, 0.0f, 0.0f);
		glUniform1f(glGetUniformLocation(litProgram, "uWaveAmount"), 0.0f);
		setCommonUniforms(litProgram, 0.0f, 0.4f);

		glBindVertexArray(sphere.vao);
		glDrawArrays(GL_TRIANGLES, 0, sphere.vertexCount);
		glBindVertexArray(0);

		glDisable(GL_BLEND);
	}

	// ДИРИЖАБЛЬ (используем загруженную OBJ модель или fallback на примитивы)
	Entity airshipRender = airship;
	glm::mat4 asModel(1.0f);
	asModel = glm::translate(asModel, airshipRender.position);
	asModel = glm::rotate(asModel, yaw, glm::vec3(0.0f, 1.0f, 0.0f));

	if (airshipModel.vertexCount > 0) {
		// Используем загруженную модель
		// Временно отключаем culling для корректного отображения
		glDisable(GL_CULL_FACE);
		glDisable(GL_BLEND);

		glm::mat4 bodyModel = glm::scale(asModel, glm::vec3(1.5f, 1.5f, 1.5f)); // Масштаб модели
		glm::mat4 mvp = cam.projection * cam.view * bodyModel;
		glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(bodyModel)));
		glUseProgram(litProgram);
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
		glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(bodyModel));
		glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
		glUniform3f(glGetUniformLocation(litProgram, "uColor"), airshipRender.color.x, airshipRender.color.y, airshipRender.color.z);
		glUniform1f(glGetUniformLocation(litProgram, "uWaveAmount"), 0.0f);
		setCommonUniforms(litProgram, 0.05f, 1.0f);
		glBindVertexArray(airshipModel.vao);
		glDrawArrays(GL_TRIANGLES, 0, airshipModel.vertexCount);
		glBindVertexArray(0);

		// Включаем culling обратно
		glEnable(GL_CULL_FACE);
		// blending остаётся выключенным для непрозрачной геометрии
	}
	else {
		// Fallback: рисуем примитивами
		// Рисуем баллон (тело)
		{
			glm::mat4 bodyModel = glm::scale(asModel, airshipRender.scale);
			glm::mat4 mvp = cam.projection * cam.view * bodyModel;
			glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(bodyModel)));
			glUseProgram(litProgram);
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(bodyModel));
			glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
			glUniform3f(glGetUniformLocation(litProgram, "uColor"), airshipRender.color.x, airshipRender.color.y, airshipRender.color.z);
			glUniform1f(glGetUniformLocation(litProgram, "uWaveAmount"), 0.0f);
			setCommonUniforms(litProgram, 0.0f, 1.0f);
			glBindVertexArray(sphere.vao);
			glDrawArrays(GL_TRIANGLES, 0, sphere.vertexCount);
		}
		// Рисуем гондолу (кабину) снизу
		{
			glm::mat4 gondolaModel = glm::translate(asModel, glm::vec3(0.0f, -0.8f, 0.0f));
			gondolaModel = glm::scale(gondolaModel, glm::vec3(0.8f, 0.5f, 0.6f));
			glm::mat4 mvp = cam.projection * cam.view * gondolaModel;
			glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(gondolaModel)));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(gondolaModel));
			glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
			glUniform3f(glGetUniformLocation(litProgram, "uColor"), 0.4f, 0.3f, 0.2f);
			glBindVertexArray(cube.vao);
			glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
		}
		// Хвостовые стабилизаторы (4 плавника)
		for (int i = 0; i < 4; ++i)
		{
			float finAngle = i * PI * 0.5f;
			glm::mat4 finModel = glm::translate(asModel, glm::vec3(0.0f, 0.0f, -1.1f));
			finModel = glm::rotate(finModel, finAngle, glm::vec3(0.0f, 0.0f, 1.0f));
			finModel = glm::translate(finModel, glm::vec3(0.0f, 0.4f, 0.0f));
			finModel = glm::scale(finModel, glm::vec3(0.08f, 0.5f, 0.4f));

			glm::mat4 mvp = cam.projection * cam.view * finModel;
			glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(finModel)));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(finModel));
			glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
			glUniform3f(glGetUniformLocation(litProgram, "uColor"), 0.7f, 0.2f, 0.2f);
			glBindVertexArray(cube.vao);
			glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
		}
		// Пропеллер сзади (вращающийся)
		{
			glm::mat4 propModel = glm::translate(asModel, glm::vec3(0.0f, 0.0f, -1.4f));
			propModel = glm::rotate(propModel, time * 15.0f, glm::vec3(0.0f, 0.0f, 1.0f));
			propModel = glm::scale(propModel, glm::vec3(0.6f, 0.08f, 0.08f));

			glm::mat4 mvp = cam.projection * cam.view * propModel;
			glm::mat3 normal = glm::mat3(glm::transpose(glm::inverse(propModel)));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
			glUniformMatrix4fv(glGetUniformLocation(litProgram, "uModel"), 1, GL_FALSE, glm::value_ptr(propModel));
			glUniformMatrix3fv(glGetUniformLocation(litProgram, "uNormalMatrix"), 1, GL_FALSE, glm::value_ptr(normal));
			glUniform3f(glGetUniformLocation(litProgram, "uColor"), 0.3f, 0.3f, 0.3f);
			glBindVertexArray(cube.vao);
			glDrawArrays(GL_TRIANGLES, 0, cube.vertexCount);
		}
		glBindVertexArray(0);
	}


	// Декорации: первые 4 — домики, следующие 4 — маленькие ёлочки
	for (size_t i = 0; i < decorations.size(); ++i)
	{
		const auto& d = decorations[i];

		if (i < 4) {
			// Домик (тип декорации 1)
			// Основание домика (стены)
			Entity walls;
			walls.position = d.position + glm::vec3(0.0f, 0.5f, 0.0f);
			walls.scale = { 1.2f, 1.0f, 1.2f };
			walls.color = { 0.9f, 0.85f, 0.7f }; // Бежевые стены
			drawMeshUnlit(cube, walls, walls.color);

			// Крыша (конус) - высоко над стенами, не пересекается
			Entity roof;
			roof.position = d.position + glm::vec3(0.0f, 1.5f, 0.0f);
			roof.scale = { 1.4f, 0.7f, 1.4f };
			roof.color = { 0.7f, 0.2f, 0.15f }; // Красная крыша
			drawMeshUnlit(cone, roof, roof.color);

			// Дверь - на передней стене
			Entity door;
			door.position = d.position + glm::vec3(0.0f, 0.35f, 0.62f);
			door.scale = { 0.3f, 0.6f, 0.05f };
			door.color = { 0.4f, 0.25f, 0.1f }; // Коричневая дверь
			drawMeshUnlit(cube, door, door.color);

			// Окна (два маленьких квадрата)
			for (int side = -1; side <= 1; side += 2)
			{
				Entity window;
				window.position = d.position + glm::vec3(side * 0.35f, 0.6f, 0.62f);
				window.scale = { 0.2f, 0.2f, 0.05f };
				window.color = { 0.6f, 0.8f, 1.0f }; // Голубое окно
				drawMeshUnlit(cube, window, window.color);
			}

			// Труба - торчит из крыши
			Entity chimney;
			chimney.position = d.position + glm::vec3(0.35f, 1.7f, 0.0f);
			chimney.scale = { 0.15f, 0.4f, 0.15f };
			chimney.color = { 0.5f, 0.3f, 0.2f }; // Коричневая труба
			drawMeshUnlit(cube, chimney, chimney.color);

		}
		else {
			// Маленькая ёлочка (тип декорации 2) с колыханием
			// Ствол (не колышется)
			Entity trunk;
			trunk.position = d.position + glm::vec3(0.0f, 0.15f, 0.0f);
			trunk.scale = { 0.12f, 0.3f, 0.12f };
			trunk.color = { 0.35f, 0.2f, 0.1f };
			drawMeshUnlit(cube, trunk, trunk.color);

			// Нижний ярус - лёгкое колыхание
			Entity layer1;
			layer1.position = d.position + glm::vec3(0.0f, 0.6f, 0.0f);
			layer1.scale = { 0.6f, 0.4f, 0.6f };
			layer1.color = { 0.1f, 0.4f, 0.15f };
			drawMeshUnlitWave(cone, layer1, layer1.color, 0.04f);

			// Верхний ярус - чуть сильнее колыхание
			Entity layer2;
			layer2.position = d.position + glm::vec3(0.0f, 1.0f, 0.0f);
			layer2.scale = { 0.4f, 0.35f, 0.4f };
			layer2.color = { 0.1f, 0.45f, 0.15f };
			drawMeshUnlitWave(cone, layer2, layer2.color, 0.06f);
		}
	}

	// Третий тип декораций: снеговики (добавим их отдельно)
	for (int i = 0; i < 3; ++i)
	{
		glm::vec3 snowmanPos = glm::vec3(
			std::cos(i * PI * 0.67f + 1.0f) * 12.0f,
			0.0f,
			std::sin(i * PI * 0.67f + 1.0f) * 12.0f
		);

		// Нижний шар
		Entity bottom;
		bottom.position = snowmanPos + glm::vec3(0.0f, 0.5f, 0.0f);
		bottom.scale = { 1.0f, 1.0f, 1.0f };
		bottom.color = { 0.95f, 0.95f, 0.98f };
		drawMeshUnlit(sphere, bottom, bottom.color);

		// Средний шар
		Entity middle;
		middle.position = snowmanPos + glm::vec3(0.0f, 1.2f, 0.0f);
		middle.scale = { 0.7f, 0.7f, 0.7f };
		middle.color = { 0.95f, 0.95f, 0.98f };
		drawMeshUnlit(sphere, middle, middle.color);

		// Голова
		Entity head;
		head.position = snowmanPos + glm::vec3(0.0f, 1.75f, 0.0f);
		head.scale = { 0.5f, 0.5f, 0.5f };
		head.color = { 0.95f, 0.95f, 0.98f };
		drawMeshUnlit(sphere, head, head.color);

		// Нос-морковка
		Entity nose;
		nose.position = snowmanPos + glm::vec3(0.0f, 1.75f, 0.28f);
		nose.scale = { 0.08f, 0.08f, 0.25f };
		nose.color = { 0.9f, 0.5f, 0.1f };
		drawMeshUnlit(cone, nose, nose.color);

		// Шляпа (цилиндр из куба)
		Entity hat;
		hat.position = snowmanPos + glm::vec3(0.0f, 2.1f, 0.0f);
		hat.scale = { 0.4f, 0.35f, 0.4f };
		hat.color = { 0.1f, 0.1f, 0.1f };
		drawMeshUnlit(cube, hat, hat.color);

		// Поля шляпы
		Entity hatBrim;
		hatBrim.position = snowmanPos + glm::vec3(0.0f, 1.95f, 0.0f);
		hatBrim.scale = { 0.6f, 0.05f, 0.6f };
		hatBrim.color = { 0.1f, 0.1f, 0.1f };
		drawMeshUnlit(cube, hatBrim, hatBrim.color);
	}

	Entity parcelEntity;
	parcelEntity.scale = { 0.4f, 0.4f, 0.4f };
	parcelEntity.color = { 0.8f, 0.6f, 0.3f };
	for (auto& p : parcels)
	{
		if (!p.active) continue;
		parcelEntity.position = p.position;
		drawMesh(sphere, parcelEntity, litProgram, 0.0f, 0.0f, 1.0f);
	}

	Entity gift;
	gift.scale = { 0.5f, 0.5f, 0.5f };
	for (size_t i = 0; i < gifts.size(); ++i)
	{
		// Чередуем цвета подарков
		const glm::vec3 giftColors[] = {
			{0.2f, 0.2f, 0.8f}, {0.8f, 0.2f, 0.2f}, {0.2f, 0.8f, 0.2f}, {0.8f, 0.8f, 0.2f}
		};
		gift.position = gifts[i];
		gift.color = giftColors[i % 4];
		drawMesh(cube, gift, litProgram, 0.0f, 0.0f, 1.0f);

		// Ленточка на подарке (вертикальная полоска)
		Entity ribbon;
		ribbon.position = gifts[i] + glm::vec3(0.0f, 0.05f, 0.0f);
		ribbon.scale = { 0.55f, 0.1f, 0.15f };
		ribbon.color = { 1.0f, 0.9f, 0.2f };
		drawMesh(cube, ribbon, litProgram, 0.0f, 0.0f, 1.0f);
	}

	// Облака с постоянным свечением (не зависят от освещения сцены)
	glEnable(GL_BLEND);
	Entity cloudEntity;
	cloudEntity.color = { 1.0f, 1.0f, 1.0f };
	bool flash = std::fmod(time, 6.0f) < 0.5f;
	for (auto& c : clouds)
	{
		cloudEntity.position = c.basePosition;
		// Облака плоские и вытянутые
		cloudEntity.scale = glm::vec3(c.radius * 2.5f, c.radius * 0.4f, c.radius * 1.8f);
		// Облака всегда немного светятся (emission 0.5), вспышка молнии добавляет яркости
		float emission = flash ? 0.8f : 0.5f;
		drawMesh(cube, cloudEntity, litProgram, 0.1f, emission, 0.7f);
	}

	for (auto& b : balloons)
	{
		// Шар
		drawMesh(sphere, b, litProgram, 0.05f, 0.0f, 0.9f);
		// Корзина под шаром
		Entity basket;
		basket.position = b.position + glm::vec3(0.0f, -0.8f, 0.0f);
		basket.scale = { 0.4f, 0.3f, 0.4f };
		basket.color = { 0.5f, 0.35f, 0.2f };
		drawMesh(cube, basket, litProgram, 0.0f, 0.0f, 1.0f);
	}

	// На всякий случай возвращаем состояние blending к выключенному
	glDisable(GL_BLEND);
}

GLuint Game::compileShader(GLenum type, const char* src) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);
	GLint status = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (!status) {
		GLint length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		std::string log(static_cast<size_t>(length), '\0');
		glGetShaderInfoLog(shader, length, nullptr, log.data());
		std::cerr << "Shader compile error: " << log << "\n";
	}
	return shader;
}

GLuint Game::linkProgram(const char* vs, const char* fs) {
	GLuint program = glCreateProgram();
	GLuint v = compileShader(GL_VERTEX_SHADER, vs);
	GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
	glAttachShader(program, v);
	glAttachShader(program, f);
	glLinkProgram(program);
	GLint status = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (!status) {
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

Mesh Game::makeCube() {
	// Все грани с CCW winding (смотрим снаружи куба)
	const float verts[] = {
		// Front face (+Z) - CCW: BL, BR, TR, BL, TR, TL
		-0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,0.0f,
		 0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,0.0f,
		 0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,1.0f,
		-0.5f,-0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,0.0f,
		 0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 1.0f,1.0f,
		-0.5f, 0.5f, 0.5f,  0.0f,0.0f,1.0f, 0.0f,1.0f,
		// Back face (-Z) - CCW from outside: BR, BL, TL, BR, TL, TR
		 0.5f,-0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,0.0f,
		-0.5f,-0.5f,-0.5f,  0.0f,0.0f,-1.0f, 1.0f,0.0f,
		-0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 1.0f,1.0f,
		 0.5f,-0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,0.0f,
		-0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 1.0f,1.0f,
		 0.5f, 0.5f,-0.5f,  0.0f,0.0f,-1.0f, 0.0f,1.0f,
		 // Left face (-X) - CCW from outside
		 -0.5f,-0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,0.0f,
		 -0.5f,-0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,0.0f,
		 -0.5f, 0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,1.0f,
		 -0.5f,-0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,0.0f,
		 -0.5f, 0.5f, 0.5f, -1.0f,0.0f,0.0f, 1.0f,1.0f,
		 -0.5f, 0.5f,-0.5f, -1.0f,0.0f,0.0f, 0.0f,1.0f,
		 // Right face (+X) - CCW from outside
		  0.5f,-0.5f, 0.5f,  1.0f,0.0f,0.0f, 0.0f,0.0f,
		  0.5f,-0.5f,-0.5f,  1.0f,0.0f,0.0f, 1.0f,0.0f,
		  0.5f, 0.5f,-0.5f,  1.0f,0.0f,0.0f, 1.0f,1.0f,
		  0.5f,-0.5f, 0.5f,  1.0f,0.0f,0.0f, 0.0f,0.0f,
		  0.5f, 0.5f,-0.5f,  1.0f,0.0f,0.0f, 1.0f,1.0f,
		  0.5f, 0.5f, 0.5f,  1.0f,0.0f,0.0f, 0.0f,1.0f,
		  // Top face (+Y) - CCW from above: TL, BL, BR, TL, BR, TR
		  -0.5f, 0.5f,-0.5f,  0.0f,1.0f,0.0f, 0.0f,0.0f,
		  -0.5f, 0.5f, 0.5f,  0.0f,1.0f,0.0f, 0.0f,1.0f,
		   0.5f, 0.5f, 0.5f,  0.0f,1.0f,0.0f, 1.0f,1.0f,
		  -0.5f, 0.5f,-0.5f,  0.0f,1.0f,0.0f, 0.0f,0.0f,
		   0.5f, 0.5f, 0.5f,  0.0f,1.0f,0.0f, 1.0f,1.0f,
		   0.5f, 0.5f,-0.5f,  0.0f,1.0f,0.0f, 1.0f,0.0f,
		   // Bottom face (-Y) - CCW from below
		   -0.5f,-0.5f, 0.5f,  0.0f,-1.0f,0.0f, 0.0f,0.0f,
		   -0.5f,-0.5f,-0.5f,  0.0f,-1.0f,0.0f, 0.0f,1.0f,
			0.5f,-0.5f,-0.5f,  0.0f,-1.0f,0.0f, 1.0f,1.0f,
		   -0.5f,-0.5f, 0.5f,  0.0f,-1.0f,0.0f, 0.0f,0.0f,
			0.5f,-0.5f,-0.5f,  0.0f,-1.0f,0.0f, 1.0f,1.0f,
			0.5f,-0.5f, 0.5f,  0.0f,-1.0f,0.0f, 1.0f,0.0f
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

Mesh Game::makeQuad() {
	const float verts[] = {
		// Triangle 1
		-0.5f, 0.0f,  0.5f,  0.0f,1.0f,0.0f,  0.0f,1.0f, // BL
		 0.5f, 0.0f,  0.5f,  0.0f,1.0f,0.0f,  1.0f,1.0f, // BR
		-0.5f, 0.0f, -0.5f,  0.0f,1.0f,0.0f,  0.0f,0.0f, // TL

		// Triangle 2
		-0.5f, 0.0f, -0.5f,  0.0f,1.0f,0.0f,  0.0f,0.0f, // TL
		 0.5f, 0.0f,  0.5f,  0.0f,1.0f,0.0f,  1.0f,1.0f, // BR
		 0.5f, 0.0f, -0.5f,  0.0f,1.0f,0.0f,  1.0f,0.0f  // TR
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

// Новая функция для создания сферы (CCW winding для корректного culling)
Mesh Game::makeSphere(int stacks, int slices) {
	std::vector<float> verts;

	auto pushVert = [&](float phi, float theta, float u, float v) {
		float px = std::sin(phi) * std::cos(theta);
		float py = std::cos(phi);
		float pz = std::sin(phi) * std::sin(theta);
		// Позиция (радиус 0.5), нормаль, UV
		verts.insert(verts.end(), { px * 0.5f, py * 0.5f, pz * 0.5f, px, py, pz, u, v });
		};

	for (int i = 0; i < stacks; ++i) {
		float v0 = static_cast<float>(i) / stacks;
		float v1 = static_cast<float>(i + 1) / stacks;
		float phi0 = v0 * PI;
		float phi1 = v1 * PI;

		for (int j = 0; j < slices; ++j) {
			float u0 = static_cast<float>(j) / slices;
			float u1 = static_cast<float>(j + 1) / slices;
			float theta0 = u0 * PI * 2.0f;
			float theta1 = u1 * PI * 2.0f;

			// Два треугольника на квад, CCW winding (смотрим снаружи)
			// Треугольник 1: P0 -> P1 -> P2
			pushVert(phi0, theta0, u0, v0); // P0 (top-left)
			pushVert(phi0, theta1, u1, v0); // P1 (top-right)
			pushVert(phi1, theta0, u0, v1); // P2 (bottom-left)

			// Треугольник 2: P1 -> P3 -> P2
			pushVert(phi0, theta1, u1, v0); // P1 (top-right)
			pushVert(phi1, theta1, u1, v1); // P3 (bottom-right)
			pushVert(phi1, theta0, u0, v1); // P2 (bottom-left)
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

// Конус для ёлки (вершина вверху, основание внизу)
Mesh Game::makeCone(int segments) {
	std::vector<float> verts;

	float height = 1.0f;
	float radius = 0.5f;

	// Вершина конуса
	glm::vec3 tip(0.0f, height * 0.5f, 0.0f);

	for (int i = 0; i < segments; ++i) {
		float theta0 = (static_cast<float>(i) / segments) * PI * 2.0f;
		float theta1 = (static_cast<float>(i + 1) / segments) * PI * 2.0f;

		// Точки на основании
		glm::vec3 p0(std::cos(theta0) * radius, -height * 0.5f, std::sin(theta0) * radius);
		glm::vec3 p1(std::cos(theta1) * radius, -height * 0.5f, std::sin(theta1) * radius);

		// Нормаль для боковой грани (наклонная)
		float ny = radius / height;
		glm::vec3 n0 = glm::normalize(glm::vec3(std::cos(theta0), ny, std::sin(theta0)));
		glm::vec3 n1 = glm::normalize(glm::vec3(std::cos(theta1), ny, std::sin(theta1)));
		glm::vec3 nTip = glm::normalize(n0 + n1);

		auto push = [&](const glm::vec3& pos, const glm::vec3& norm, float u, float v) {
			verts.insert(verts.end(), { pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, u, v });
			};

		// Боковая грань (CCW снаружи): p0 -> p1 -> tip
		push(p0, n0, static_cast<float>(i) / segments, 0.0f);
		push(p1, n1, static_cast<float>(i + 1) / segments, 0.0f);
		push(tip, nTip, (static_cast<float>(i) + 0.5f) / segments, 1.0f);

		// Нижняя грань (основание) - нормаль вниз
		glm::vec3 downNorm(0.0f, -1.0f, 0.0f);
		glm::vec3 center(0.0f, -height * 0.5f, 0.0f);

		// CCW снизу: center -> p1 -> p0
		push(center, downNorm, 0.5f, 0.5f);
		push(p1, downNorm, 0.5f + std::cos(theta1) * 0.5f, 0.5f + std::sin(theta1) * 0.5f);
		push(p0, downNorm, 0.5f + std::cos(theta0) * 0.5f, 0.5f + std::sin(theta0) * 0.5f);
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

Mesh Game::makeTerrain(int grid, float scale, const std::vector<float>& heights) {
	std::vector<float> verts;
	verts.reserve((grid - 1) * (grid - 1) * 6 * 8);
	auto idx = [grid](int x, int z) { return z * grid + x; };
	auto heightAt = [&](int x, int z) { return heights[idx(std::clamp(x, 0, grid - 1), std::clamp(z, 0, grid - 1))]; };

	for (int z = 0; z < grid - 1; ++z) {
		for (int x = 0; x < grid - 1; ++x) {
			glm::vec3 p0 = { (x - grid / 2) * scale, heightAt(x, z), (z - grid / 2) * scale };         // TL
			glm::vec3 p1 = { (x + 1 - grid / 2) * scale, heightAt(x + 1, z), (z - grid / 2) * scale };     // TR
			glm::vec3 p2 = { (x + 1 - grid / 2) * scale, heightAt(x + 1, z + 1), (z + 1 - grid / 2) * scale }; // BR
			glm::vec3 p3 = { (x - grid / 2) * scale, heightAt(x, z + 1), (z + 1 - grid / 2) * scale };     // BL

			glm::vec3 n0 = glm::normalize(glm::cross(p3 - p0, p1 - p0));
			glm::vec3 n1 = glm::normalize(glm::cross(p3 - p1, p2 - p1));

			auto pushVert = [&](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
				verts.insert(verts.end(), { p.x, p.y, p.z, n.x, n.y, n.z, uv.x, uv.y });
				};

			pushVert(p0, n0, { 0.0f, 0.0f });
			pushVert(p3, n0, { 0.0f, 1.0f });
			pushVert(p1, n0, { 1.0f, 0.0f });

			pushVert(p1, n1, { 1.0f, 0.0f });
			pushVert(p3, n1, { 0.0f, 1.0f });
			pushVert(p2, n1, { 1.0f, 1.0f });
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

float Game::sampleHeight(const std::vector<float>& h, int grid, float x, float z) {
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

bool Game::loadHeightmap(const std::string& path, int grid, float amplitude, std::vector<float>& heightsOut) {
	sf::Image img;
	if (!img.loadFromFile(path)) return false;

	heightsOut.resize(grid * grid);
	auto sample = [&](float u, float v) {
		unsigned x = static_cast<unsigned>(u * (img.getSize().x - 1));
		unsigned y = static_cast<unsigned>(v * (img.getSize().y - 1));
		sf::Color c = img.getPixel(x, y);
		float g = static_cast<float>(c.r) / 255.0f;
		return (g - 0.5f) * amplitude;
		};

	for (int z = 0; z < grid; ++z) {
		for (int x = 0; x < grid; ++x) {
			float u = x / static_cast<float>(grid - 1);
			float v = z / static_cast<float>(grid - 1);
			heightsOut[z * grid + x] = sample(u, v);
		}
	}
	return true;
}

glm::mat4 Game::composeModel(const Entity& e) {
	glm::mat4 model(1.0f);
	model = glm::translate(model, e.position);
	model = glm::scale(model, e.scale);
	return model;
}

float Game::randFloat(float min, float max) {
	static std::mt19937 rng(static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	std::uniform_real_distribution<float> dist(min, max);
	return dist(rng);
}

glm::vec3 Game::randomXZ(float range) {
	return { randFloat(-range, range), 0.0f, randFloat(-range, range) };
}

bool Game::overlaps(const Entity& a, const Entity& b) {
	return glm::length(a.position - b.position) < (a.radius + b.radius);
}

void Game::respawnTarget(Entity& target, const std::vector<Entity>& all, float groundOffset) {
	bool good = false;
	int tries = 0;
	while (!good && tries < 64) {
		++tries;
		target.position = randomXZ(25.0f);
		target.position.y = groundOffset;
		good = true;
		for (const auto& other : all) {
			if (&other != &target && overlaps(target, other)) {
				good = false;
				break;
			}
		}
	}
}

Mesh Game::loadOBJ(const std::string& filename) {
	tinyobj::ObjReader reader;
	tinyobj::ObjReaderConfig config;
	config.triangulate = true;

	if (!reader.ParseFromFile(filename, config)) {
		std::cerr << "Failed to load OBJ file: " << filename << std::endl;
		if (!reader.Error().empty()) {
			std::cerr << "Error: " << reader.Error() << std::endl;
		}
		return Mesh{};
	}

	if (!reader.Warning().empty()) {
		std::cerr << "OBJ warning: " << reader.Warning() << std::endl;
	}

	const auto& attrib = reader.GetAttrib();
	const auto& shapes = reader.GetShapes();

	std::vector<float> verts;

	for (const auto& shape : shapes) {
		for (size_t f = 0; f < shape.mesh.indices.size(); ++f) {
			const auto& idx = shape.mesh.indices[f];

			// Position
			float vx = attrib.vertices[3 * idx.vertex_index + 0];
			float vy = attrib.vertices[3 * idx.vertex_index + 1];
			float vz = attrib.vertices[3 * idx.vertex_index + 2];

			// Normal
			float nx = 0.0f, ny = 1.0f, nz = 0.0f;
			if (idx.normal_index >= 0 && static_cast<size_t>(idx.normal_index * 3 + 2) < attrib.normals.size()) {
				nx = attrib.normals[3 * idx.normal_index + 0];
				ny = attrib.normals[3 * idx.normal_index + 1];
				nz = attrib.normals[3 * idx.normal_index + 2];
			}

			// Texcoord
			float tx = 0.0f, ty = 0.0f;
			if (idx.texcoord_index >= 0 && static_cast<size_t>(idx.texcoord_index * 2 + 1) < attrib.texcoords.size()) {
				tx = attrib.texcoords[2 * idx.texcoord_index + 0];
				ty = attrib.texcoords[2 * idx.texcoord_index + 1];
			}

			verts.insert(verts.end(), { vx, vy, vz, nx, ny, nz, tx, ty });
		}
	}

	if (verts.empty()) {
		std::cerr << "OBJ file is empty or has no valid geometry: " << filename << std::endl;
		return Mesh{};
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

	std::cout << "Loaded OBJ: " << filename << " with " << m.vertexCount << " vertices" << std::endl;
	return m;
}

int main() {
	Game game;
	if (!game.init()) return -1;
	game.run();
	return 0;
}