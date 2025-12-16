// =============================================================
// Солнечная система на OpenGL + SFML
// Библиотеки:
//  - SFML 2.6.1
//  - glad (OpenGL 3.3 Core)
//  - glm 1.0.2
//  - tinyobjloader v2.0.0
// =============================================================

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics/Image.hpp>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// ---------------- Шейдеры ----------------
const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

void main() {
    TexCoord = aTex;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D tex;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float useLighting; // 0.0 = Sun (unlit), 1.0 = Planet (lit)

void main() {
    vec4 texColor = texture(tex, TexCoord);
    
    if (useLighting < 0.5) {
        FragColor = texColor; // Sun is self-luminous
    } else {
        // Ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * vec3(1.0);

        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0);

        // Specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * vec3(1.0);

        vec3 result = (ambient + diffuse + specular) * texColor.rgb;
        FragColor = vec4(result, texColor.a);
    }
}
)";

// ---------------- Камера ----------------
struct Camera {
    glm::vec3 pos{ 0.0f, 5.0f, 20.0f };
    glm::vec3 front{ 0.0f, 0.0f, -1.0f };
    glm::vec3 up{ 0.0f, 1.0f, 0.0f };
    glm::vec3 right{ 1.0f, 0.0f, 0.0f };
    glm::vec3 worldUp{ 0.0f, 1.0f, 0.0f };

    float yaw{ -90.0f };
    float pitch{ -20.0f };
    float speed{ 10.0f };
    float sensitivity{ 0.1f };

    Camera() { updateVectors(); }

    glm::mat4 view() const {
        return glm::lookAt(pos, pos + front, up);
    }

    void processMouse(float xoffset, float yoffset) {
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        updateVectors();
    }

    void updateVectors() {
        glm::vec3 f;
        f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        f.y = sin(glm::radians(pitch));
        f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(f);
        right = glm::normalize(glm::cross(front, worldUp));
        up = glm::normalize(glm::cross(right, front));
    }
};

// ---------------- Утилиты ----------------
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    // Check errors (omitted for brevity but recommended)
    return s;
}

GLuint createProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

bool loadOBJ(const std::string& path, std::vector<float>& data, int& verts) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
        std::cerr << "OBJ Load Error: " << warn << err << std::endl;
        return false;
    }

    for (auto& s : shapes) {
        for (auto& i : s.mesh.indices) {
            // Position
            data.push_back(attrib.vertices[3 * i.vertex_index + 0]);
            data.push_back(attrib.vertices[3 * i.vertex_index + 1]);
            data.push_back(attrib.vertices[3 * i.vertex_index + 2]);

            // Normal
            if (i.normal_index >= 0) {
                data.push_back(attrib.normals[3 * i.normal_index + 0]);
                data.push_back(attrib.normals[3 * i.normal_index + 1]);
                data.push_back(attrib.normals[3 * i.normal_index + 2]);
            } else {
                data.push_back(0.0f); data.push_back(1.0f); data.push_back(0.0f);
            }

            // TexCoord
            if (i.texcoord_index >= 0) {
                data.push_back(attrib.texcoords[2 * i.texcoord_index + 0]);
                data.push_back(1.0f - attrib.texcoords[2 * i.texcoord_index + 1]);
            } else {
                data.push_back(0.0f); data.push_back(0.0f);
            }
        }
    }
    verts = data.size() / 8; // 3 pos + 3 norm + 2 tex = 8 floats
    return true;
}

GLuint loadTexture(const std::string& path) {
    sf::Image img;
    if (!img.loadFromFile(path)) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        // Return a dummy texture or handle error
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.getSize().x, img.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, img.getPixelsPtr());
    glGenerateMipmap(GL_TEXTURE_2D);

    return tex;
}

// ---------------- main ----------------
int main() {
    sf::ContextSettings s;
    s.depthBits = 24;
    s.majorVersion = 3;
    s.minorVersion = 3;
    s.attributeFlags = sf::ContextSettings::Core;

    sf::Window win({ 1280,720 }, "Improved Solar System", sf::Style::Default, s);
    win.setMouseCursorGrabbed(true);
    win.setMouseCursorVisible(false);
    win.setVerticalSyncEnabled(true);

    if (!gladLoadGL()) {
        std::cerr << "Failed to init GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    GLuint prog = createProgram();

    // ---- Модель ----
    std::vector<float> mesh;
    int vertCount = 0;
    if (!loadOBJ("Ornament.obj", mesh, vertCount)) {
        std::cerr << "Failed to load model.obj" << std::endl;
        // Fallback cube or exit? Let's just exit or continue with empty mesh
    }

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(float), mesh.data(), GL_STATIC_DRAW);

    // Stride = 8 * sizeof(float)
    // Pos (0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal (1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Tex (2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    GLuint tex = loadTexture("texture.png");

    Camera cam;
    sf::Clock clock;
    float totalTime = 0.0f;

    struct Planet { float dist, size, speed, spin; };
    std::vector<Planet> planets = {
        {0.0f, 3.0f, 0.0f, 0.3f},   // Sun
        {6.0f, 1.0f, 0.6f, 1.5f},   // Planet 1
        {9.0f, 0.7f, 0.4f, 1.2f},   // Planet 2
        {13.0f, 1.2f, 0.3f, 0.8f},  // Planet 3
        {18.0f, 0.9f, 0.2f, 1.0f},  // Planet 4
    };

    bool firstMouse = true;
    sf::Vector2i lastMousePos = sf::Mouse::getPosition(win);

    while (win.isOpen()) {
        float dt = clock.restart().asSeconds();
        totalTime += dt;

        sf::Event e;
        while (win.pollEvent(e)) {
            if (e.type == sf::Event::Closed) win.close();
            if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) win.close();
            if (e.type == sf::Event::Resized) glViewport(0, 0, e.size.width, e.size.height);
        }

        // Camera Input
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) cam.pos += cam.front * cam.speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) cam.pos -= cam.front * cam.speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) cam.pos -= cam.right * cam.speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) cam.pos += cam.right * cam.speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) cam.pos += cam.worldUp * cam.speed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) cam.pos -= cam.worldUp * cam.speed * dt;

        // Mouse Input
        sf::Vector2i mousePos = sf::Mouse::getPosition(win);
        if (firstMouse) {
            lastMousePos = mousePos;
            firstMouse = false;
        }
        float xoffset = (float)(mousePos.x - lastMousePos.x);
        float yoffset = (float)(lastMousePos.y - mousePos.y); // reversed y
        lastMousePos = mousePos;
        cam.processMouse(xoffset, yoffset);
        
        // Keep mouse centered if grabbed (optional, but good for FPS)
        // sf::Mouse::setPosition(sf::Vector2i(win.getSize().x/2, win.getSize().y/2), win);
        // lastMousePos = sf::Vector2i(win.getSize().x/2, win.getSize().y/2);
        // (Skipping centering for simplicity, standard relative movement works fine if cursor is hidden)

        glClearColor(0.05f, 0.05f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, tex);

        glm::mat4 proj = glm::perspective(glm::radians(60.f), (float)win.getSize().x / win.getSize().y, 0.1f, 100.f);
        glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, glm::value_ptr(cam.view()));
        
        // Lighting Uniforms
        glUniform3f(glGetUniformLocation(prog, "lightPos"), 0.0f, 0.0f, 0.0f); // Sun is at 0,0,0
        glUniform3f(glGetUniformLocation(prog, "viewPos"), cam.pos.x, cam.pos.y, cam.pos.z);

        for (size_t i = 0; i < planets.size(); ++i) {
            const auto& p = planets[i];
            glm::mat4 m(1.0f);
            
            // Orbit
            if (p.dist > 0.0f) {
                m = glm::rotate(m, totalTime * p.speed, { 0,1,0 });
                m = glm::translate(m, { p.dist,0,0 });
            }
            // Spin
            m = glm::rotate(m, totalTime * p.spin, { 0,1,0 });
            
            float modelScale = 0.3f; 
            m = glm::scale(m, glm::vec3(p.size * modelScale));

            glUniformMatrix4fv(glGetUniformLocation(prog, "model"), 1, GL_FALSE, glm::value_ptr(m));
            
            // Is Sun? (Index 0)
            glUniform1f(glGetUniformLocation(prog, "useLighting"), (i == 0) ? 0.0f : 1.0f);

            glDrawArrays(GL_TRIANGLES, 0, vertCount);
        }

        win.display();
    }
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(prog);
    
    return 0;
}
