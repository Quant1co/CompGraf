#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <SFML/Window/Keyboard.hpp>

Camera::Camera(glm::vec3 startPos)
{
    position = startPos;
    front = glm::vec3(0, 0, -1);
    up = glm::vec3(0, 1, 0);

    speed = 5.0f;
    yaw = -90.f;
    pitch = 0.f;
}

void Camera::update(float dt)
{
    float velocity = speed * dt;

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        position += front * velocity;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        position -= front * velocity;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        position -= glm::normalize(glm::cross(front, up)) * velocity;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        position += glm::normalize(glm::cross(front, up)) * velocity;
}

glm::mat4 Camera::getView() const
{
    return glm::lookAt(position, position + front, up);
}

void Camera::processMouse(float dx, float dy)
{
    float sens = 0.1f;
    yaw += dx * sens;
    pitch -= dy * sens;

    pitch = glm::clamp(pitch, -89.f, 89.f);

    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(dir);
}

