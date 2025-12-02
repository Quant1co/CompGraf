#define _USE_MATH_DEFINES // Для M_PI в MSVC

#include <SFML/Window.hpp>
#include <glad/glad.h>
#include <iostream>
#include <vector>
#include <cmath>

// Выбери режим закрашивания (раскомментируй один)
// #define MODE_FLAT_CONSTANT  // Задание 2: плоское константой в шейдере
// #define MODE_FLAT_UNIFORM   // Задание 3: плоское через uniform
 #define MODE_GRADIENT       // Задание 4: градиент (каждая вершина своим цветом)

// Вершинный шейдер
const char* vertexShaderSource =
#if defined(MODE_GRADIENT)
R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
out vec4 vertexColor;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    vertexColor = aColor;
}
)";
#else
R"(
#version 330 core
layout (location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)";
#endif

// Фрагментный шейдер
const char* fragmentShaderSource =
#if defined(MODE_GRADIENT)
R"(
#version 330 core
out vec4 FragColor;
in vec4 vertexColor;
void main() {
    FragColor = vertexColor;
}
)";
#elif defined(MODE_FLAT_UNIFORM)
R"(
#version 330 core
out vec4 FragColor;
uniform vec4 ourColor;
void main() {
    FragColor = ourColor;
}
)";
#elif defined(MODE_FLAT_CONSTANT)
R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); // Оранжевый для задания 2
}
)";
#else
R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f); // Белый по умолчанию (для задания 1)
}
)";
#endif

// Функция для компиляции шейдера
unsigned int compileShader(const char* source, GLenum type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
    }
    return shader;
}

// Функция для создания программы шейдера
unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    unsigned int fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Program linking error: " << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
}

// Генерация вершин для четырехугольника
std::vector<float> getQuadVertices(float offsetX = -0.6f) {
    float size = 0.2f;
#if defined(MODE_GRADIENT)
    // С цветами: нижний левый - красный, нижний правый - зеленый, верхний левый - синий, верхний правый - желтый (дубли для треугольников)
    std::vector<float> vertices;
    // Первый треугольник
    vertices.push_back(offsetX - size); vertices.push_back(-size); vertices.push_back(0.0f);
    vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);  // Красный
    vertices.push_back(offsetX + size); vertices.push_back(-size); vertices.push_back(0.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);  // Зеленый
    vertices.push_back(offsetX - size); vertices.push_back(size); vertices.push_back(0.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);  // Синий
    // Второй треугольник
    vertices.push_back(offsetX + size); vertices.push_back(-size); vertices.push_back(0.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);  // Зеленый
    vertices.push_back(offsetX + size); vertices.push_back(size); vertices.push_back(0.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);  // Желтый
    vertices.push_back(offsetX - size); vertices.push_back(size); vertices.push_back(0.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);  // Синий
    return vertices;
#else
    return {
        offsetX - size, -size, 0.0f,
        offsetX + size, -size, 0.0f,
        offsetX - size,  size, 0.0f,
        offsetX + size, -size, 0.0f,
        offsetX + size,  size, 0.0f,
        offsetX - size,  size, 0.0f
    };
#endif
}

// Генерация вершин для веера
std::vector<float> getFanVertices(float offsetX = 0.0f) {
    std::vector<float> vertices;
    float centerX = offsetX, centerY = 0.0f;
    int segments = 32;
    float radius = 0.2f;

    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * static_cast<float>(M_PI) * i / segments;
        float angle2 = 2.0f * static_cast<float>(M_PI) * (i + 1) / segments;

#if defined(MODE_GRADIENT)
        // Центр - белый, внешние - градиент по радуге
        float colorR1 = (sin(angle1) + 1.0f) / 2.0f;
        float colorG1 = (cos(angle1) + 1.0f) / 2.0f;
        float colorB1 = (sin(angle1 + 2.0f) + 1.0f) / 2.0f;
        float colorR2 = (sin(angle2) + 1.0f) / 2.0f;
        float colorG2 = (cos(angle2) + 1.0f) / 2.0f;
        float colorB2 = (sin(angle2 + 2.0f) + 1.0f) / 2.0f;

        vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(0.0f);
        vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f); // Центр белый
        vertices.push_back(centerX + radius * cos(angle1)); vertices.push_back(centerY + radius * sin(angle1)); vertices.push_back(0.0f);
        vertices.push_back(colorR1); vertices.push_back(colorG1); vertices.push_back(colorB1); vertices.push_back(1.0f);
        vertices.push_back(centerX + radius * cos(angle2)); vertices.push_back(centerY + radius * sin(angle2)); vertices.push_back(0.0f);
        vertices.push_back(colorR2); vertices.push_back(colorG2); vertices.push_back(colorB2); vertices.push_back(1.0f);
#else
        vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(0.0f);
        vertices.push_back(centerX + radius * cos(angle1)); vertices.push_back(centerY + radius * sin(angle1)); vertices.push_back(0.0f);
        vertices.push_back(centerX + radius * cos(angle2)); vertices.push_back(centerY + radius * sin(angle2)); vertices.push_back(0.0f);
#endif
    }
    return vertices;
}

// Генерация вершин для пятиугольника
std::vector<float> getPentagonVertices(float offsetX = 0.6f) {
    std::vector<float> vertices;
    float centerX = offsetX, centerY = 0.0f;
    int sides = 5;
    float radius = 0.2f;
    float startAngle = static_cast<float>(M_PI) / 2.0f;

    for (int i = 0; i < sides; ++i) {
        float angle1 = startAngle - 2.0f * static_cast<float>(M_PI) * i / sides;
        float angle2 = startAngle - 2.0f * static_cast<float>(M_PI) * (i + 1) / sides;

#if defined(MODE_GRADIENT)
        // Центр - белый, вершины - разные цвета
        float colors[5][3] = { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f} };

        vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(0.0f);
        vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f); // Центр белый
        vertices.push_back(centerX + radius * cos(angle1)); vertices.push_back(centerY + radius * sin(angle1)); vertices.push_back(0.0f);
        vertices.push_back(colors[i % 5][0]); vertices.push_back(colors[i % 5][1]); vertices.push_back(colors[i % 5][2]); vertices.push_back(1.0f);
        vertices.push_back(centerX + radius * cos(angle2)); vertices.push_back(centerY + radius * sin(angle2)); vertices.push_back(0.0f);
        vertices.push_back(colors[(i + 1) % 5][0]); vertices.push_back(colors[(i + 1) % 5][1]); vertices.push_back(colors[(i + 1) % 5][2]); vertices.push_back(1.0f);
#else
        vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(0.0f);
        vertices.push_back(centerX + radius * cos(angle1)); vertices.push_back(centerY + radius * sin(angle1)); vertices.push_back(0.0f);
        vertices.push_back(centerX + radius * cos(angle2)); vertices.push_back(centerY + radius * sin(angle2)); vertices.push_back(0.0f);
#endif
    }
    return vertices;
}

int main() {
    sf::Window window(sf::VideoMode(800, 600), "OpenGL Lab 11", sf::Style::Default, sf::ContextSettings(24, 8, 0, 3, 3));

    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    unsigned int shaderProgram = createShaderProgram();

    unsigned int VAOs[3], VBOs[3];
    glGenVertexArrays(3, VAOs);
    glGenBuffers(3, VBOs);

    auto quadVertices = getQuadVertices(-0.6f);
    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, quadVertices.size() * sizeof(float), quadVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
#if defined(MODE_GRADIENT)
        7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
#else
        3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
#endif

    auto fanVertices = getFanVertices(0.0f);
    glBindVertexArray(VAOs[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
    glBufferData(GL_ARRAY_BUFFER, fanVertices.size() * sizeof(float), fanVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
#if defined(MODE_GRADIENT)
        7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
#else
        3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
#endif

    auto pentagonVertices = getPentagonVertices(0.6f);
    glBindVertexArray(VAOs[2]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[2]);
    glBufferData(GL_ARRAY_BUFFER, pentagonVertices.size() * sizeof(float), pentagonVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
#if defined(MODE_GRADIENT)
        7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
#else
        3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
#endif

    // Для uniform: location
#if defined(MODE_FLAT_UNIFORM)
    int colorLocation = glGetUniformLocation(shaderProgram, "ourColor");
#endif

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

#if defined(MODE_FLAT_UNIFORM)
        // Пример: разный цвет для каждой фигуры (передача перед draw)
        glUniform4f(colorLocation, 1.0f, 0.0f, 0.0f, 1.0f); // Красный для квадрата
#endif
        glBindVertexArray(VAOs[0]);
        glDrawArrays(GL_TRIANGLES, 0, quadVertices.size() /
#if defined(MODE_GRADIENT)
            7
#else
            3
#endif
        );

#if defined(MODE_FLAT_UNIFORM)
        glUniform4f(colorLocation, 0.0f, 1.0f, 0.0f, 1.0f); // Зеленый для веера
#endif
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_TRIANGLES, 0, fanVertices.size() /
#if defined(MODE_GRADIENT)
            7
#else
            3
#endif
        );

#if defined(MODE_FLAT_UNIFORM)
        glUniform4f(colorLocation, 0.0f, 0.0f, 1.0f, 1.0f); // Синий для пятиугольника
#endif
        glBindVertexArray(VAOs[2]);
        glDrawArrays(GL_TRIANGLES, 0, pentagonVertices.size() /
#if defined(MODE_GRADIENT)
            7
#else
            3
#endif
        );

        window.display();
    }

    glDeleteVertexArrays(3, VAOs);
    glDeleteBuffers(3, VBOs);
    glDeleteProgram(shaderProgram);

    return 0;
}