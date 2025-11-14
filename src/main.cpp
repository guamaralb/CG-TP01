// main.cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std::chrono;

// ---------- Configurações ----------
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

struct Boid {
    glm::vec3 pos;
    glm::vec3 vel;
    float wingState; // para animar asas
};

std::vector<Boid> boids;
Boid targetBoid;
bool paused = false;
int cameraMode = 1; // 1 = topo da torre, 2 = atrás do bando, 3 = perpendicular
float worldSize = 200.0f;

// parâmetros do comportamento
float separationRadius = 5.0f;
float neighborRadius = 20.0f;
float maxSpeed = 15.0f;
float maxForce = 5.0f;
float separationWeight = 1.5f;
float cohesionWeight = 1.0f;
float alignmentWeight = 1.0f;
int initialBoids = 60;

// Temporização
double lastTime = 0.0;

// util
std::mt19937 rng((unsigned)steady_clock::now().time_since_epoch().count());
std::uniform_real_distribution<float> unif(-1.0f, 1.0f);

float randf(float a, float b) {
    return a + (b - a) * ((unif(rng) + 1.0f) / 2.0f);
}

// ---------- Shaders (simples) ----------
const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(aPos,1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
)glsl";

const char* fragmentShaderSrc = R"glsl(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 baseColor;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);

    // simple ambient + diffuse + spec
    float ambient = 0.25;
    float spec = 0.0;
    if(diff > 0.0) {
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.5;
    }
    vec3 color = (ambient + diff) * baseColor + spec * vec3(1.0);
    FragColor = vec4(color, 1.0);
}
)glsl";

// compile util
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}

GLuint createProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    int ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(prog, 1024, nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ---------- Geometrias: pirâmide (boid), cone (tower), plane ----------
struct Mesh {
    GLuint VAO = 0, VBO = 0, EBO = 0;
    GLsizei count = 0;
};

Mesh createBirdBoid() {
    // Vértices aproximando um pássaro 3D leve (corpo + asas + cauda)
    struct Vertex { glm::vec3 pos; glm::vec3 norm; };
    std::vector<Vertex> verts;

    // Corpo central (fino, pontudo na frente)
    glm::vec3 nose(0.0f, 0.0f, 0.6f);
    glm::vec3 bodyBack(0.0f, 0.0f, -0.6f);
    glm::vec3 bodyTop(0.0f, 0.1f, -0.3f);
    glm::vec3 bodyBottom(0.0f, -0.1f, -0.3f);

    // Asas
    glm::vec3 leftWing(-0.7f, 0.05f, -0.1f);
    glm::vec3 rightWing(0.7f, 0.05f, -0.1f);

    // Cauda (duas pequenas “penas”)
    glm::vec3 tailLeft(-0.15f, 0.0f, -0.8f);
    glm::vec3 tailRight(0.15f, 0.0f, -0.8f);

    auto addTri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        verts.push_back({ a, n });
        verts.push_back({ b, n });
        verts.push_back({ c, n });
        };

    // Corpo principal (duas faces)
    addTri(nose, bodyTop, bodyBottom);
    addTri(bodyBack, bodyBottom, bodyTop);

    // Asas (superfície superior e inferior)
    addTri(nose, leftWing, bodyTop);
    addTri(nose, bodyTop, rightWing);
    addTri(bodyBottom, leftWing, nose);
    addTri(bodyBottom, nose, rightWing);

    // Cauda
    addTri(bodyBack, tailLeft, tailRight);

    Mesh m;
    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);
    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
    m.count = (GLsizei)verts.size();
    glBindVertexArray(0);
    return m;
}

Mesh createPyramid() {
    // pyramid: tip + square base -> we'll do a 4-sided pyramid (5 vertices)
    // positions and normals approximate
    std::vector<float> data; // pos(3) normal(3)
    std::vector<unsigned int> idx;

    // vertices
    glm::vec3 tip(0.0f, 0.6f, 0.0f);
    glm::vec3 b0(0.4f, -0.4f, 0.3f);
    glm::vec3 b1(-0.4f, -0.4f, 0.3f);
    glm::vec3 b2(-0.4f, -0.4f, -0.3f);
    glm::vec3 b3(0.4f, -0.4f, -0.3f);
    std::vector<glm::vec3> verts = { tip, b0, b1, b2, b3 };

    // faces (triangles): 4 side faces + base (two triangles)
    auto pushVert = [&](glm::vec3 p, glm::vec3 n) {
        data.push_back(p.x); data.push_back(p.y); data.push_back(p.z);
        data.push_back(n.x); data.push_back(n.y); data.push_back(n.z);
        };

    auto faceNormal = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        return glm::normalize(glm::cross(b - a, c - a));
        };

    // side faces: (tip, bi, b(i+1))
    for (int i = 1; i <= 4; i++) {
        glm::vec3 a = verts[0];
        glm::vec3 b = verts[i];
        glm::vec3 c = verts[i == 4 ? 1 : i + 1];
        glm::vec3 n = faceNormal(a, b, c);
        pushVert(a, n); pushVert(b, n); pushVert(c, n);
        int base = (int)idx.size();
        idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
    }
    // base two triangles: b1,b2,b3,b0 order (we'll use center)
    glm::vec3 nBase = glm::vec3(0.0f, -1.0f, 0.0f);
    pushVert(b0, nBase); pushVert(b1, nBase); pushVert(b2, nBase); pushVert(b3, nBase);
    int baseIdx = (int)idx.size();
    idx.push_back(baseIdx); idx.push_back(baseIdx + 1); idx.push_back(baseIdx + 2);
    idx.push_back(baseIdx); idx.push_back(baseIdx + 2); idx.push_back(baseIdx + 3);

    // Actually data duplicates vertices per-face (simpler). Build VAO
    Mesh m;
    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);
    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
    // no EBO because we used sequential idx earlier, but we didn't store indices properly.
    // Simpler: draw with glDrawArrays, each triangle is 3 vertices. count = data_vertices
    m.count = (GLsizei)(data.size() / 6);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(0);
    return m;
}

Mesh createPlane() {
    float s = 400.0f;
    // positions and normals
    float data[] = {
        -s, 0.0f, -s,  0,1,0,
         s, 0.0f, -s,  0,1,0,
         s, 0.0f,  s,  0,1,0,
        -s, 0.0f,  s,  0,1,0
    };
    unsigned int idx[] = { 0,1,2, 0,2,3 };
    Mesh m;
    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);
    glGenBuffers(1, &m.EBO);
    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    m.count = 6;
    glBindVertexArray(0);
    return m;
}

// naive cone mesh for tower
Mesh createCone(int slices = 32) {
    std::vector<float> data;
    std::vector<unsigned int> idx;
    float height = 6.0f;
    float radius = 3.0f;
    glm::vec3 tip(0.0f, height, 0.0f);
    // side triangles
    for (int i = 0; i < slices; i++) {
        float a0 = (2.0f * M_PI * i) / slices;
        float a1 = (2.0f * M_PI * (i + 1)) / slices;
        glm::vec3 p0(radius * cos(a0), 0.0f, radius * sin(a0));
        glm::vec3 p1(radius * cos(a1), 0.0f, radius * sin(a1));
        glm::vec3 n = glm::normalize(glm::cross(p0 - tip, p1 - tip));
        // push triangle (tip, p0, p1)
        auto push = [&](glm::vec3 p, glm::vec3 normal) {
            data.push_back(p.x); data.push_back(p.y); data.push_back(p.z);
            data.push_back(normal.x); data.push_back(normal.y); data.push_back(normal.z);
            };
        push(tip, n); push(p0, n); push(p1, n);
    }
    // base disk
    glm::vec3 nbase(0, -1, 0);
    for (int i = 0; i < slices - 2; i++) {
        // triangle between p0, p(i+1), p(i+2)
        float a0 = 0;
        float a1 = (2.0f * M_PI * (i + 1)) / slices;
        float a2 = (2.0f * M_PI * (i + 2)) / slices;
        glm::vec3 p0(radius * cos(a0), 0, radius * sin(a0));
        glm::vec3 p1(radius * cos(a1), 0, radius * sin(a1));
        glm::vec3 p2(radius * cos(a2), 0, radius * sin(a2));
        auto push = [&](glm::vec3 p, glm::vec3 normal) {
            data.push_back(p.x); data.push_back(p.y); data.push_back(p.z);
            data.push_back(normal.x); data.push_back(normal.y); data.push_back(normal.z);
            };
        push(p0, nbase); push(p1, nbase); push(p2, nbase);
    }

    Mesh m;
    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);
    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
    m.count = (GLsizei)(data.size() / 6);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(0);
    return m;
}

// ---------- Boid rules ----------
glm::vec3 limit(const glm::vec3& v, float maxLen) {
    float l = glm::length(v);
    if (l > maxLen) return glm::normalize(v) * maxLen;
    return v;
}

void initBoids(int n) {
    boids.clear();
    boids.reserve(n);
    for (int i = 0; i < n; i++) {
        Boid b;
        b.pos = glm::vec3(randf(-20, 20), randf(1, 10), randf(-20, 20));
        glm::vec3 dir(randf(-1, 1), randf(-0.2f, 0.2f), randf(-1, 1));
        b.vel = glm::normalize(dir) * randf(2.0f, 6.0f);
        b.wingState = randf(0, 2 * M_PI);
        boids.push_back(b);
    }
    targetBoid.pos = glm::vec3(0.0f, 6.0f, 0.0f);
    targetBoid.vel = glm::vec3(0.0f, 0.0f, 5.0f);
}

glm::vec3 computeSeparation(int i) {
    glm::vec3 steer(0);
    int count = 0;
    for (int j = 0; j < (int)boids.size(); j++) {
        if (j == i) continue;
        float d = glm::distance(boids[i].pos, boids[j].pos);
        if (d < separationRadius && d>0.001f) {
            glm::vec3 diff = boids[i].pos - boids[j].pos;
            diff = glm::normalize(diff) / d; // stronger when closer
            steer += diff;
            count++;
        }
    }
    if (count > 0) steer /= (float)count;
    if (glm::length(steer) > 0.001f) {
        steer = glm::normalize(steer) * maxSpeed - boids[i].vel;
        steer = limit(steer, maxForce);
    }
    return steer;
}

glm::vec3 computeAlignment(int i) {
    glm::vec3 sum(0);
    int count = 0;
    for (int j = 0; j < (int)boids.size(); j++) {
        if (j == i) continue;
        float d = glm::distance(boids[i].pos, boids[j].pos);
        if (d < neighborRadius) {
            sum += boids[j].vel;
            count++;
        }
    }
    if (count > 0) {
        sum /= (float)count;
        sum = glm::normalize(sum) * maxSpeed;
        glm::vec3 steer = sum - boids[i].vel;
        steer = limit(steer, maxForce);
        return steer;
    }
    return glm::vec3(0.0f);
}

glm::vec3 computeCohesion(int i) {
    glm::vec3 center(0);
    int count = 0;
    for (int j = 0; j < (int)boids.size(); j++) {
        if (j == i) continue;
        float d = glm::distance(boids[i].pos, boids[j].pos);
        if (d < neighborRadius) {
            center += boids[j].pos;
            count++;
        }
    }
    if (count > 0) {
        center /= (float)count;
        glm::vec3 desired = center - boids[i].pos;
        if (glm::length(desired) > 0.001f) {
            desired = glm::normalize(desired) * maxSpeed;
            glm::vec3 steer = desired - boids[i].vel;
            steer = limit(steer, maxForce);
            return steer;
        }
    }
    return glm::vec3(0.0f);
}

glm::vec3 seekTarget(const Boid& b, const Boid& target) {
    glm::vec3 desired = target.pos - b.pos;
    float d = glm::length(desired);
    if (d < 0.001f) return glm::vec3(0.0f);
    desired = glm::normalize(desired) * maxSpeed;
    glm::vec3 steer = desired - b.vel;
    return limit(steer, maxForce);
}

void updateBoids(float dt) {
    // update target position (simple movement)
    targetBoid.pos += targetBoid.vel * dt;

    // keep target inside world bounds (wrap)
    if (targetBoid.pos.x > worldSize) targetBoid.pos.x = -worldSize;
    if (targetBoid.pos.x < -worldSize) targetBoid.pos.x = worldSize;
    if (targetBoid.pos.z > worldSize) targetBoid.pos.z = -worldSize;
    if (targetBoid.pos.z < -worldSize) targetBoid.pos.z = worldSize;

    std::vector<glm::vec3> newVel(boids.size());
    for (size_t i = 0; i < boids.size(); i++) {
        Boid& b = boids[i];
        glm::vec3 s = computeSeparation(i) * separationWeight;
        glm::vec3 a = computeAlignment(i) * alignmentWeight;
        glm::vec3 c = computeCohesion(i) * cohesionWeight;
        glm::vec3 t = seekTarget(b, targetBoid) * 0.6f;

        glm::vec3 accel = s + a + c + t;

        // avoid tower at origin (cone)
        glm::vec3 towerPos(0.0f, 0.0f, 0.0f);
        float towerRadius = 6.0f;
        glm::vec3 diff = b.pos - towerPos;
        float d = glm::length(glm::vec3(diff.x, 0.0f, diff.z));
        if (d < towerRadius + 3.0f) {
            glm::vec3 away = glm::normalize(glm::vec3(diff.x, 0.0f, diff.z)) * maxForce * 5.0f;
            accel += glm::vec3(away.x, 0.5f * away.y, away.z);
        }

        // update velocity and position
        glm::vec3 v = b.vel + accel * dt;
        v = limit(v, maxSpeed);
        // simple ground & height constraints
        if (b.pos.y < 1.0f) v.y += (1.0f - b.pos.y) * 0.5f;
        if (b.pos.y > 60.0f) v.y -= (b.pos.y - 60.0f) * 0.5f;

        newVel[i] = v;
    }
    for (size_t i = 0; i < boids.size(); i++) {
        boids[i].vel = newVel[i];
        boids[i].pos += boids[i].vel * dt;

        // wrap world in x,z
        if (boids[i].pos.x > worldSize) boids[i].pos.x = -worldSize;
        if (boids[i].pos.x < -worldSize) boids[i].pos.x = worldSize;
        if (boids[i].pos.z > worldSize) boids[i].pos.z = -worldSize;
        if (boids[i].pos.z < -worldSize) boids[i].pos.z = worldSize;

        // wing animation
        boids[i].wingState += dt * (1.0f + glm::length(boids[i].vel) * 0.1f);
    }
}

// ---------- Input handling ----------
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
    static double lastPlus = 0.0;
    double now = glfwGetTime();
    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
        if (now - lastPlus > 0.2) {
            // add boid near target
            Boid b;
            b.pos = targetBoid.pos + glm::vec3(randf(-5, 5), randf(-1, 2), randf(-5, 5));
            glm::vec3 dir(randf(-1, 1), randf(-0.2f, 0.2f), randf(-1, 1));
            b.vel = glm::normalize(dir) * randf(1.0f, 6.0f);
            b.wingState = 0;
            boids.push_back(b);
            lastPlus = now;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
        if (now - lastPlus > 0.2 && !boids.empty()) {
            boids.erase(boids.begin() + (rng() % boids.size()));
            lastPlus = now;
        }
    }
    // pause
    static bool lastSpace = false;
    bool spacePressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
    if (spacePressed && !lastSpace) paused = !paused;
    lastSpace = spacePressed;

    // camera mode
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) cameraMode = 1;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) cameraMode = 2;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) cameraMode = 3;

    // target control: arrows
    float tAccel = 6.0f;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) targetBoid.vel += glm::vec3(0, 0, -tAccel) * 0.02f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) targetBoid.vel += glm::vec3(0, 0, tAccel) * 0.02f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) targetBoid.vel += glm::vec3(-tAccel, 0, 0) * 0.02f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) targetBoid.vel += glm::vec3(tAccel, 0, 0) * 0.02f;
    if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS) targetBoid.vel *= 1.01f;
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) targetBoid.vel *= 0.99f;
}

// ---------- Render helpers ----------
void setMat4(GLuint prog, const std::string& name, const glm::mat4& m) {
    GLint loc = glGetUniformLocation(prog, name.c_str());
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}
void setVec3(GLuint prog, const std::string& name, const glm::vec3& v) {
    GLint loc = glGetUniformLocation(prog, name.c_str());
    glUniform3fv(loc, 1, glm::value_ptr(v));
}

// -------------------------------------------
// Projeção simples no plano Y=0 (para sombras)
// -------------------------------------------
glm::mat4 shadowMatrix(const glm::vec3& lightDir) {
    glm::mat4 m(1.0f);
    float lx = lightDir.x;
    float ly = lightDir.y;
    float lz = lightDir.z;

    // projeção no plano Y=0
    m[0][1] = -lx / ly;
    m[1][1] = 0.0f;
    m[2][1] = -lz / ly;
    return m;
}


// ---------- Main ----------
int main() {

    // GLFW init
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef _WIN32
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Boids - GLAD/GLFW/OpenGL", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n"; return -1;
    }
    glEnable(GL_DEPTH_TEST);

    GLuint prog = createProgram();

    // create meshes
    Mesh bird = createBirdBoid();
    Mesh plane = createPlane();
    Mesh cone = createCone(48);

    // init boids
    initBoids(initialBoids);

    lastTime = glfwGetTime();

    // main loop
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;

        processInput(window);
        if (!paused) updateBoids(dt);

        // clear
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        glClearColor(0.55f, 0.8f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);

        // camera setup based on cameraMode
        glm::vec3 lookAt = glm::vec3(0.0f);
        if (!boids.empty()) {
            // center of boid flock
            glm::vec3 mid(0.0f);
            for (auto& b : boids) mid += b.pos;
            mid /= (float)boids.size();
            lookAt = mid;
        }
        else {
            lookAt = targetBoid.pos;
        }

        glm::vec3 camPos;
        if (cameraMode == 1) { // top of tower
            camPos = glm::vec3(0.0f, 30.0f, 0.1f);
        }
        else if (cameraMode == 2) { // behind the flock
            glm::vec3 avgVel(0.0f);
            for (auto& b : boids) avgVel += b.vel;
            if (!boids.empty()) avgVel /= (float)boids.size();
            glm::vec3 dir = glm::normalize(glm::vec3(avgVel.x, 0.0f, avgVel.z) + glm::vec3(0.0001f));
            camPos = lookAt - dir * 35.0f + glm::vec3(0.0f, 12.0f, 0.0f);
        }
        else { // perpendicular
            glm::vec3 avgVel(0.0f);
            for (auto& b : boids) avgVel += b.vel;
            if (!boids.empty()) avgVel /= (float)boids.size();
            glm::vec3 dir = glm::normalize(glm::vec3(-avgVel.z, 0.0f, avgVel.x));
            camPos = lookAt + dir * 35.0f + glm::vec3(0.0f, 12.0f, 0.0f);
        }

        glm::mat4 view = glm::lookAt(camPos, lookAt, glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);

        setMat4(prog, "view", view);
        setMat4(prog, "proj", proj);
        setVec3(prog, "lightPos", glm::vec3(30.0f, 80.0f, 40.0f));
        setVec3(prog, "viewPos", camPos);

        // draw plane
        glm::mat4 model = glm::mat4(1.0f);
        setMat4(prog, "model", model);
        setVec3(prog, "baseColor", glm::vec3(0.6f, 0.75f, 0.55f));
        glBindVertexArray(plane.VAO);
        glDrawElements(GL_TRIANGLES, plane.count, GL_UNSIGNED_INT, 0);

        // ----- Sombras -----
        glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, -2.0f, 0.5f));
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        for (auto& b : boids) {
            // Direção e rotação básica do boid
            glm::vec3 dir = glm::normalize(b.vel + glm::vec3(0.0001f));
            glm::vec3 up(0, 1, 0);
            glm::vec3 right = glm::normalize(glm::cross(up, dir));
            glm::vec3 newUp = glm::normalize(glm::cross(dir, right));

            glm::mat4 baseRot(1.0f);
            baseRot[0] = glm::vec4(right, 0.0f);
            baseRot[1] = glm::vec4(newUp, 0.0f);
            baseRot[2] = glm::vec4(dir, 0.0f);

            // Estado da asa (mantém o batimento)
            float wingAngle = std::sin(b.wingState * 4.0f) * glm::radians(25.0f);
            glm::mat4 wingRot = glm::rotate(glm::mat4(1.0f), wingAngle, glm::vec3(0.0f, 0.0f, 1.0f));

            // Posição da sombra: mesmo X e Z, Y = 0.01 (levemente acima do chão)
            glm::vec3 shadowPos = glm::vec3(b.pos.x, 0.01f, b.pos.z);

            // Inclinação da sombra de acordo com a direção da luz
            glm::mat4 tilt = glm::mat4(1.0f);
            tilt[1][0] = -lightDir.x / lightDir.y;
            tilt[1][2] = -lightDir.z / lightDir.y;

            glm::mat4 model =
                glm::translate(glm::mat4(1.0f), shadowPos)
                * tilt
                * baseRot
                * wingRot
                * glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 0.001f, 1.0f)); // achata a sombra

            setMat4(prog, "model", model);
            setVec3(prog, "baseColor", glm::vec3(0.0f, 0.0f, 0.0f));

            glBindVertexArray(bird.VAO);
            glDrawArrays(GL_TRIANGLES, 0, bird.count);
        }

        glDisable(GL_BLEND);

        // draw tower (cone)
        model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        setMat4(prog, "model", model);
        setVec3(prog, "baseColor", glm::vec3(0.75f, 0.5f, 0.4f));
        glBindVertexArray(cone.VAO);
        glDrawArrays(GL_TRIANGLES, 0, cone.count);

        // draw target boid
        model = glm::translate(glm::mat4(1.0f), targetBoid.pos);
        model = glm::scale(model, glm::vec3(1.4f));
        setMat4(prog, "model", model);
        setVec3(prog, "baseColor", glm::vec3(1.0f, 0.2f, 0.2f));
        glBindVertexArray(bird.VAO);
        glDrawArrays(GL_TRIANGLES, 0, bird.count);

        // draw boids
        for (auto& b : boids) {
            // Direção e rotação básica
            glm::vec3 dir = glm::normalize(b.vel + glm::vec3(0.0001f));
            glm::vec3 up(0, 1, 0);
            glm::vec3 right = glm::normalize(glm::cross(up, dir));
            glm::vec3 newUp = glm::normalize(glm::cross(dir, right));

            glm::mat4 baseRot(1.0f);
            baseRot[0] = glm::vec4(right, 0.0f);
            baseRot[1] = glm::vec4(newUp, 0.0f);
            baseRot[2] = glm::vec4(dir, 0.0f);

            // Batimento de asas (rotação em torno do eixo do corpo)
            float wingAngle = std::sin(b.wingState * 4.0f) * glm::radians(25.0f); // amplitude de 25°
            glm::mat4 wingRot = glm::rotate(glm::mat4(1.0f), wingAngle, glm::vec3(0.0f, 0.0f, 1.0f));

            // Combina tudo (rot + rotação das asas + posição)
            glm::mat4 model = glm::translate(glm::mat4(1.0f), b.pos)
                * baseRot
                * wingRot
                * glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));

            setMat4(prog, "model", model);
            setVec3(prog, "baseColor", glm::vec3(0.9f, 0.85f, 0.6f));
            glBindVertexArray(bird.VAO);
            glDrawArrays(GL_TRIANGLES, 0, bird.count);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
