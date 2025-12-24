#pragma once
#include <glm/glm.hpp>

class Camera
{
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;

    float speed;
    float yaw;
    float pitch;

    Camera(glm::vec3 startPos);

    void update(float dt);
    glm::mat4 getView() const;
    void processMouse(float dx, float dy);

};
#pragma once
