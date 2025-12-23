#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>

struct Mesh
{
    GLuint vao{};
    GLuint vbo{};
    GLsizei vertexCount{};
};

struct Entity
{
    glm::vec3 position{ 0.0f };
    glm::vec3 scale{ 1.0f };
    glm::vec3 color{ 1.0f };
    float radius{ 1.0f };
    bool visible{ true };
};

struct Parcel
{
    glm::vec3 position{ 0.0f };
    glm::vec3 velocity{ 0.0f };
    float radius{ 0.3f };
    bool active{ false };
};

struct Cloud
{
    glm::vec3 basePosition{ 0.0f };
    float radius{ 2.0f };
    float phase{ 0.0f };
};

struct Camera
{
    glm::vec3 position{ 0.0f };
    glm::vec3 target{ 0.0f };
    glm::mat4 view{ 1.0f };
    glm::mat4 projection{ 1.0f };
};

class Game
{
public:
    Game();
    ~Game();
    bool init();
    void run();

private:
    void setupScene();
    void processEvents();
    void update(float dt, float time);
    void render(float time);
    void cleanup();

    GLuint compileShader(GLenum type, const char* src);
    GLuint linkProgram(const char* vs, const char* fs);

    Mesh makeCube();
    Mesh makeQuad();
    Mesh makeSphere(int stacks, int slices);
    Mesh makeCone(int segments); // Конус для ёлки
    Mesh makeTerrain(int grid, float scale, const std::vector<float>& heights);
    bool loadHeightmap(const std::string& path, int grid, float amplitude, std::vector<float>& heightsOut);
    Mesh loadOBJ(const std::string& filename);
    float sampleHeight(const std::vector<float>& h, int grid, float x, float z);
    glm::mat4 composeModel(const Entity& e);
    float randFloat(float min, float max);
    glm::vec3 randomXZ(float range);
    bool overlaps(const Entity& a, const Entity& b);
    void respawnTarget(Entity& target, const std::vector<Entity>& all, float groundOffset);

private:
    sf::RenderWindow window;
    GLuint litProgram{};
    GLuint targetProgram{};

    Mesh cube{};
    Mesh quad{};
    Mesh sphere{};
    Mesh cone{};
    Mesh terrain{};
    Mesh airshipModel{};
    Mesh treeModel{};
    std::vector<float> terrainHeights;
    int terrainGrid = 64;
    float terrainScale = 1.0f;

    Entity airship{};
    glm::vec3 airshipVelocity{ 0.0f };

    std::vector<Entity> targets;
    std::vector<Entity> decorations;
    std::vector<Cloud> clouds;
    std::vector<Entity> balloons;
    std::vector<Entity> lamps;
    Entity sleigh{};
    float sleighAngle = 0.0f;
    float sleighRadius = 3.5f;
    float sleighSpeed = 0.8f;
    std::vector<Parcel> parcels;
    std::vector<glm::vec3> gifts;

    glm::vec3 treePos{ 0.0f, 0.0f, -5.0f };

    int parcelIndex = 0;
    int score = 0;
    bool spotlightOn = true;
    float yaw = 0.0f;
    bool aimCamera = false;
    sf::Clock clock;
};