#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Shader.h"

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 tex;
};

class Model
{
public:
    Model(const std::string& objPath, const std::string& texturePath);
    void draw(const Shader& shader,
        const glm::mat4& modelMatrix,
        int shadingModel);

private:
    unsigned int VAO, VBO;
    unsigned int textureID;
    int vertexCount;

    void loadOBJ(const std::string& path);
    unsigned int loadTexture(const std::string& path);
};
#pragma once
