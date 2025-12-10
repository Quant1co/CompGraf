#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp> // Для текстур
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

out vec3 ourColor;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float colorInfluence; // Для задания 2
uniform float textureMix;     // Для задания 3

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSourceTask1 = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(ourColor, 1.0);
}
)";

const char* fragmentShaderSourceTask2 = R"(
#version 330 core
in vec3 ourColor;
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ourTexture;
uniform float colorInfluence;

void main() {
    vec4 texColor = texture(ourTexture, TexCoord);
    FragColor = mix(texColor, vec4(ourColor, 1.0), colorInfluence);
}
)";

const char* fragmentShaderSourceTask3 = R"(
#version 330 core
in vec3 ourColor;
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float textureMix;

void main() {
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), textureMix);
}
)";

const char* fragmentShaderSourceTask4 = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(ourColor, 1.0);
}
)";

// Функция для компиляции шейдера
GLuint compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
    }
    return shader;
}

// Функция для создания программы шейдера
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Program linking error: " << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

int main() {
    // Инициализация окна SFML
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 4;
    settings.majorVersion = 3;
    settings.minorVersion = 3;
    sf::Window window(sf::VideoMode(1200, 800), "OpenGL Lab 12", sf::Style::Default, settings);
    window.setFramerateLimit(60);

    // Инициализация GLAD
    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // Создание шейдерных программ для каждого задания
    GLuint shaderProgramTask1 = createShaderProgram(vertexShaderSource, fragmentShaderSourceTask1);
    GLuint shaderProgramTask2 = createShaderProgram(vertexShaderSource, fragmentShaderSourceTask2);
    GLuint shaderProgramTask3 = createShaderProgram(vertexShaderSource, fragmentShaderSourceTask3);
    GLuint shaderProgramTask4 = createShaderProgram(vertexShaderSource, fragmentShaderSourceTask4);

    // --- Задание 1: Градиентный тетраэдр ---
    // Вершины тетраэдра с скорректированными цветами (верх blue, левый red, правый green, задний yellow)
    std::vector<float> tetrahedronVertices = {
        // Позиции          Цвета             Текстурные координаты (не используются)
         0.0f,  0.5f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // Вершина 0: blue
        -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // Вершина 1: red
         0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, // Вершина 2: green
         0.0f, -0.5f,  0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f  // Вершина 3: yellow
    };
    std::vector<unsigned int> tetrahedronIndices = {
        0, 1, 2,
        0, 1, 3,
        0, 2, 3,
        1, 2, 3
    };

    GLuint VAO1, VBO1, EBO1;
    glGenVertexArrays(1, &VAO1);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &EBO1);

    glBindVertexArray(VAO1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, tetrahedronVertices.size() * sizeof(float), tetrahedronVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO1);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, tetrahedronIndices.size() * sizeof(unsigned int), tetrahedronIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glm::vec3 tetraPos(-2.0f, 0.0f, 0.0f); // Позиция тетраэдра

    // --- Задание 2: Кубик с текстурой и цветом ---
    std::vector<float> cubeVertices = {
        // Позиции          Цвета             Текстурные координаты
        -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f
    };
    std::vector<unsigned int> cubeIndices = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 1, 5, 5, 4, 0,
        2, 3, 7, 7, 6, 2,
        3, 0, 4, 4, 7, 3,
        1, 2, 6, 6, 5, 1
    };

    GLuint VAO2, VBO2, EBO2;
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &EBO2);

    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, cubeVertices.size() * sizeof(float), cubeVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cubeIndices.size() * sizeof(unsigned int), cubeIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Загрузка текстуры для задания 2
    sf::Image textureImg1;
    if (!textureImg1.loadFromFile("texture1.png")) {
        std::cerr << "Failed to load texture1.png" << std::endl;
        return -1;
    }
    GLuint texture1;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImg1.getSize().x, textureImg1.getSize().y, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureImg1.getPixelsPtr());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float colorInfluence = 0.5f; // Влияние цвета для задания 2

    // --- Задание 3: Кубик с двумя текстурами ---
    // Используем те же вершины и индексы, что в задании 2
    GLuint VAO3, VBO3, EBO3;
    glGenVertexArrays(1, &VAO3);
    glGenBuffers(1, &VBO3);
    glGenBuffers(1, &EBO3);

    glBindVertexArray(VAO3);
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, cubeVertices.size() * sizeof(float), cubeVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cubeIndices.size() * sizeof(unsigned int), cubeIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Вторая текстура
    sf::Image textureImg2;
    if (!textureImg2.loadFromFile("texture2.png")) {
        std::cerr << "Failed to load texture2.png" << std::endl;
        return -1;
    }
    GLuint texture2;
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureImg2.getSize().x, textureImg2.getSize().y, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureImg2.getPixelsPtr());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float textureMix = 0.5f; // Микс текстур для задания 3

    // --- Задание 4: Градиентный круг ---
    // Генерация диска (центр белый, окружность HSV Hue)
    const int segments = 100;
    std::vector<float> circleVertices;
    std::vector<unsigned int> circleIndices;

    // Центр
    circleVertices.push_back(0.0f); circleVertices.push_back(0.0f); circleVertices.push_back(0.0f); // Pos
    circleVertices.push_back(1.0f); circleVertices.push_back(1.0f); circleVertices.push_back(1.0f); // Color white
    circleVertices.push_back(0.0f); circleVertices.push_back(0.0f); // Tex (не исп)

    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * glm::pi<float>() * i / segments;
        float x = 0.5f * cos(angle);
        float y = 0.5f * sin(angle);
        // HSV to RGB: Hue = angle / (2*pi), S=1, V=1
        float h = angle / (2.0f * glm::pi<float>());
        float r = abs(h * 6.0f - 3.0f) - 1.0f;
        float g = 2.0f - abs(h * 6.0f - 2.0f);
        float b = 2.0f - abs(h * 6.0f - 4.0f);
        r = glm::clamp(r, 0.0f, 1.0f);
        g = glm::clamp(g, 0.0f, 1.0f);
        b = glm::clamp(b, 0.0f, 1.0f);

        circleVertices.push_back(x); circleVertices.push_back(y); circleVertices.push_back(0.0f);
        circleVertices.push_back(r); circleVertices.push_back(g); circleVertices.push_back(b);
        circleVertices.push_back(0.0f); circleVertices.push_back(0.0f);

        if (i < segments) {
            circleIndices.push_back(0);
            circleIndices.push_back(i + 1);
            circleIndices.push_back(i + 2);
        }
    }

    GLuint VAO4, VBO4, EBO4;
    glGenVertexArrays(1, &VAO4);
    glGenBuffers(1, &VBO4);
    glGenBuffers(1, &EBO4);

    glBindVertexArray(VAO4);
    glBindBuffer(GL_ARRAY_BUFFER, VBO4);
    glBufferData(GL_ARRAY_BUFFER, circleVertices.size() * sizeof(float), circleVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO4);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, circleIndices.size() * sizeof(unsigned int), circleIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glm::vec3 circleScale(1.0f, 1.0f, 1.0f); // Масштаб по осям для задания 4

    // Матрицы
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1200.0f / 800.0f, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // Основной цикл
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed) {
                // Управление тетраэдром (Задание 1): W/A/S/D для XY, Q/E для Z
                if (event.key.code == sf::Keyboard::W) tetraPos.y += 0.1f;
                if (event.key.code == sf::Keyboard::S) tetraPos.y -= 0.1f;
                if (event.key.code == sf::Keyboard::A) tetraPos.x -= 0.1f;
                if (event.key.code == sf::Keyboard::D) tetraPos.x += 0.1f;
                if (event.key.code == sf::Keyboard::Q) tetraPos.z += 0.1f;
                if (event.key.code == sf::Keyboard::E) tetraPos.z -= 0.1f;

                // Влияние цвета (Задание 2): Up/Down
                if (event.key.code == sf::Keyboard::Up) colorInfluence = glm::min(1.0f, colorInfluence + 0.1f);
                if (event.key.code == sf::Keyboard::Down) colorInfluence = glm::max(0.0f, colorInfluence - 0.1f);

                // Микс текстур (Задание 3): Left/Right
                if (event.key.code == sf::Keyboard::Left) textureMix = glm::max(0.0f, textureMix - 0.1f);
                if (event.key.code == sf::Keyboard::Right) textureMix = glm::min(1.0f, textureMix + 0.1f);

                // Масштаб круга (Задание 4): X: Z/X, Y: C/V, Z: B/N
                if (event.key.code == sf::Keyboard::Z) circleScale.x += 0.1f;
                if (event.key.code == sf::Keyboard::X) circleScale.x -= 0.1f;
                if (event.key.code == sf::Keyboard::C) circleScale.y += 0.1f;
                if (event.key.code == sf::Keyboard::V) circleScale.y -= 0.1f;
                if (event.key.code == sf::Keyboard::B) circleScale.z += 0.1f;
                if (event.key.code == sf::Keyboard::N) circleScale.z -= 0.1f;
            }
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- Рендер Задание 1: Тетраэдр ---
        glUseProgram(shaderProgramTask1);
        glm::mat4 model1 = glm::translate(glm::mat4(1.0f), tetraPos);
        model1 = glm::rotate(model1, glm::radians(30.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // Лёгкий наклон по X для 3D-вида (меньше 45 для плоскости)
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask1, "model"), 1, GL_FALSE, glm::value_ptr(model1));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask1, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask1, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(VAO1);
        glDrawElements(GL_TRIANGLES, tetrahedronIndices.size(), GL_UNSIGNED_INT, 0);

        // --- Рендер Задание 2: Куб с текстурой и цветом ---
        glUseProgram(shaderProgramTask2);
        glm::mat4 model2 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.7f, 0.0f, 0.0f));
        model2 = glm::rotate(model2, (float)glm::radians(sf::Clock().getElapsedTime().asSeconds() * 50.0f), glm::vec3(0.5f, 1.0f, 0.0f)); // Легкий поворот для динамики
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask2, "model"), 1, GL_FALSE, glm::value_ptr(model2));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask2, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask2, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(shaderProgramTask2, "colorInfluence"), colorInfluence);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glUniform1i(glGetUniformLocation(shaderProgramTask2, "ourTexture"), 0);
        glBindVertexArray(VAO2);
        glDrawElements(GL_TRIANGLES, cubeIndices.size(), GL_UNSIGNED_INT, 0);

        // --- Рендер Задание 3: Куб с двумя текстурами ---
        glUseProgram(shaderProgramTask3);
        glm::mat4 model3 = glm::translate(glm::mat4(1.0f), glm::vec3(0.7f, 0.0f, 0.0f));
        model3 = glm::rotate(model3, (float)glm::radians(sf::Clock().getElapsedTime().asSeconds() * -50.0f), glm::vec3(0.5f, 1.0f, 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask3, "model"), 1, GL_FALSE, glm::value_ptr(model3));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask3, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask3, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(shaderProgramTask3, "textureMix"), textureMix);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glUniform1i(glGetUniformLocation(shaderProgramTask3, "texture1"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
        glUniform1i(glGetUniformLocation(shaderProgramTask3, "texture2"), 1);
        glBindVertexArray(VAO3);
        glDrawElements(GL_TRIANGLES, cubeIndices.size(), GL_UNSIGNED_INT, 0);

        // --- Рендер Задание 4: Градиентный круг ---
        glUseProgram(shaderProgramTask4);
        glm::mat4 model4 = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
        model4 = glm::scale(model4, circleScale);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask4, "model"), 1, GL_FALSE, glm::value_ptr(model4));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask4, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramTask4, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(VAO4);
        glDrawElements(GL_TRIANGLES, circleIndices.size(), GL_UNSIGNED_INT, 0);

        window.display();
    }

    // Очистка
    glDeleteVertexArrays(1, &VAO1); glDeleteBuffers(1, &VBO1); glDeleteBuffers(1, &EBO1);
    glDeleteVertexArrays(1, &VAO2); glDeleteBuffers(1, &VBO2); glDeleteBuffers(1, &EBO2);
    glDeleteVertexArrays(1, &VAO3); glDeleteBuffers(1, &VBO3); glDeleteBuffers(1, &EBO3);
    glDeleteVertexArrays(1, &VAO4); glDeleteBuffers(1, &VBO4); glDeleteBuffers(1, &EBO4);
    glDeleteProgram(shaderProgramTask1);
    glDeleteProgram(shaderProgramTask2);
    glDeleteProgram(shaderProgramTask3);
    glDeleteProgram(shaderProgramTask4);
    glDeleteTextures(1, &texture1);
    glDeleteTextures(1, &texture2);

    return 0;
}