#include <SFML/Window.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Shader.h"
#include "Model.h"
#include "Camera.h"

int main()
{
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.majorVersion = 3;
    settings.minorVersion = 3;
    settings.attributeFlags = sf::ContextSettings::Core;

    sf::Window window(sf::VideoMode(1280, 720), "Lab 14 - Aligned Objects",
        sf::Style::Default, settings);
    window.setVerticalSyncEnabled(true);

    gladLoadGL();
    glEnable(GL_DEPTH_TEST);

    Shader shader("vertex.glsl", "fragment.glsl");
    // Камеру чуть отодвинем и поднимем, чтобы видеть все сверху
    Camera camera(glm::vec3(0.0f, 4.0f, 10.0f));

    // ===== ЗАГРУЗКА МОДЕЛЕЙ =====
    Model plane("plane.obj", "yellow.jpg");

    Model barrel("barrel.obj", "barrel.jpg");
    Model bottle("bottle.obj", "bottle.jpg");
    Model pyramid("pyramid.obj", "pyramid.jpg");
    Model cube("cube.obj", "cube.jpg");
    Model cone("cone.obj", "cone.jpg");

    sf::Clock clock;
    bool flashlightOn = true;

    while (window.isOpen())
    {
        static bool first = true;
        static sf::Vector2i last;

        sf::Vector2i cur = sf::Mouse::getPosition(window);
        if (!first)
            camera.processMouse(cur.x - last.x, cur.y - last.y);
        first = false;
        last = cur;

        sf::Event e;
        while (window.pollEvent(e))
        {
            if (e.type == sf::Event::Closed) window.close();
            if (e.type == sf::Event::KeyPressed)
            {
                if (e.key.code == sf::Keyboard::F) flashlightOn = !flashlightOn;
                if (e.key.code == sf::Keyboard::Escape) window.close();
            }
        }

        float dt = clock.restart().asSeconds();
        camera.update(dt);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        shader.setMat4("view", camera.getView());
        shader.setMat4("projection", glm::perspective(glm::radians(60.f), 1280.f / 720.f, 0.1f, 100.f));
        shader.setVec3("viewPos", camera.position);

        // ===== СВЕТ =====
        shader.setVec3("dirLight.direction", { -0.2f, -1.0f, -0.3f });
        shader.setVec3("dirLight.color", { 1.0f, 1.0f, 1.0f });

        shader.setVec3("pointLight.position", { 0.0f, 3.0f, 2.0f });
        shader.setVec3("pointLight.color", { 1.0f, 0.8f, 0.6f });
        shader.setFloat("pointLight.intensity", 1.0f);
        shader.setFloat("pointLight.constant", 1.0f);
        shader.setFloat("pointLight.linear", 0.09f);
        shader.setFloat("pointLight.quadratic", 0.032f);

        shader.setVec3("spotLight.color", flashlightOn ? glm::vec3(1.0f) : glm::vec3(0.0f));
        shader.setVec3("spotLight.position", camera.position);
        shader.setVec3("spotLight.direction", camera.front);
        shader.setFloat("spotLight.cutOff", glm::cos(glm::radians(12.5f)));
        shader.setFloat("spotLight.outerCutOff", glm::cos(glm::radians(17.5f)));

        // ===== ОТРИСОВКА ОБЪЕКТОВ =====

        // 1. ПОЛ (Scale x10, позиция Y=0)
        glm::mat4 planeModel = glm::mat4(1.0f);
        planeModel = glm::translate(planeModel, glm::vec3(0.0f, 0.0f, 0.0f));
        planeModel = glm::scale(planeModel, glm::vec3(10.0f, 1.0f, 10.0f));
        plane.draw(shader, planeModel, 0);

        // Расставляем объекты в линию по оси X (слева направо).
        // Y = 1.0f для объектов, у которых центр в середине (Куб, Бочка).
        // Y = 0.0f для объектов, у которых центр в основании (обычно Конус).

        // Бочка (слева)
        // Если бочка "утонет", поменяйте Y на 1.0f
        barrel.draw(shader, glm::translate(glm::mat4(1), { -4.0f, 1.3f, 1.0f }), 0);

        // Конус (слева-центр)
        // Конусы часто имеют origin внизу, поэтому Y=0
        cone.draw(shader, glm::translate(glm::mat4(1), { -2.0f, 0.0f, 0.0f }), 0);

        // Бутылка (центр)
        bottle.draw(shader, glm::translate(glm::mat4(1), { 0.0f, 1.3f, 0.0f }), 1);

        // Пирамида (справа-центр)
        // Если пирамида летает, поставьте Y=0.0f, если тонет — верните 1.0f
        pyramid.draw(shader, glm::translate(glm::mat4(1), { 2.0f, 0.0f, 0.0f }), 2);

        // Куб (справа)
        // Куб обычно 2x2x2 с центром в (0,0,0), поэтому поднимаем на 1
        cube.draw(shader, glm::translate(glm::mat4(1), { 4.0f, 0.5f, 0.0f }), 1);

        window.display();
    }
}