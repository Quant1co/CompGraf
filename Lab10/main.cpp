#include <SFML/Window.hpp>
#include <glad/glad.h>  // GLAD для OpenGL
#include <iostream>
#include <vector>
#include <optional>  // Для std::optional в SFML 3.0

// Функция для логирования ошибок шейдера
void ShaderLog(GLuint shader) {
    GLint infologLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1) {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
        std::cout << "Shader InfoLog: " << infoLog.data() << std::endl;
    }
}

// Функция для проверки ошибок OpenGL
void CheckOpenGLError(const std::string& action) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error after " << action << ": " << err << std::endl;
    }
}

int main() {
    // Настройки контекста
    sf::ContextSettings settings;
    settings.depthBits = 24u;
    settings.stencilBits = 8u;
    //settings.antialiasingLevel = 0u;  // Если ошибка здесь, закомментируй строку и попробуй
    settings.majorVersion = 3u;
    settings.minorVersion = 3u;

    // Создаём окно (explicit Vector2u)
    //sf::Window window(sf::VideoMode(sf::Vector2u{ 800u, 600u }), "Green Triangle", sf::Style::Default, settings);
    //window.setActive(true);

    // Если ошибка в конструкторе, попробуй эту версию без settings:
    sf::Window window(sf::VideoMode(sf::Vector2u{800u, 600u}), "Green Triangle");

    // Инициализируем GLAD
    if (!gladLoadGL()) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Устанавливаем viewport
    glViewport(0, 0, 800, 600);

    // Устанавливаем чёрный фон
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    CheckOpenGLError("glClearColor");

    // 1. Инициализируем шейдеры
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main() {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    ShaderLog(vertexShader);
    CheckOpenGLError("Compile vertex shader");

    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0);  // Зелёный
        }
    )";

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    ShaderLog(fragmentShader);
    CheckOpenGLError("Compile fragment shader");

    // 2. Инициализируем шейдерную программу
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    CheckOpenGLError("Link program");

    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        std::vector<char> infoLog(512);
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog.data());
        std::cout << "Program link error: " << infoLog.data() << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // 3. Получаем ID атрибута
    GLint vertexPosAttrib = glGetAttribLocation(shaderProgram, "aPos");
    if (vertexPosAttrib == -1) {
        std::cout << "Error: Attribute 'aPos' not found" << std::endl;
        return -1;
    }

    // 4. Инициализируем VBO
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    CheckOpenGLError("Initialize VBO");
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Основной цикл
    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        // 5. Отрисовываем сцену
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glEnableVertexAttribArray(vertexPosAttrib);
        glVertexAttribPointer(vertexPosAttrib, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

        glDrawArrays(GL_TRIANGLES, 0, 3);
        CheckOpenGLError("Draw");

        glDisableVertexAttribArray(vertexPosAttrib);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

        window.display();
    }

    // 6. Очищаем ресурсы
    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &VBO);

    return 0;
}