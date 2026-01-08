#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

struct Camera {
	float fov = 45.0f;
	float near = 0.1f;
	float far = 100.0f;

    Camera(glm::vec3 pos, float x_angle, float y_angle) : position(pos), x_angle(x_angle), y_angle(y_angle) {
        calculate_front();
    }

	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up = {0, 1, 0};

    float x_angle, y_angle;
    inline void rotX(float x) { x_angle = glm::mod(x_angle + x, 360.0f); calculate_front(); }
    inline void rotY(float y) { y_angle = glm::clamp(y_angle + y, -89.0f, 89.0f); calculate_front(); }
    inline void rotXY(float x, float y) { 
        x_angle = glm::mod(x_angle + x, 360.0f);
        y_angle = glm::clamp(y_angle + y, -89.0f, 89.0f);
        calculate_front();
    }
    void calculate_front() {
        front = glm::normalize(glm::vec3{
            glm::cos(glm::radians(x_angle)) * glm::cos(glm::radians(y_angle)),
            glm::sin(glm::radians(y_angle)),
            glm::sin(glm::radians(x_angle)) * glm::cos(glm::radians(y_angle))
        });
    }

    glm::vec3 screenPos(int w, int h, glm::vec3 coord) {
        glm::mat4 perspective = glm::perspective(
            glm::radians(fov), // perspective 
            (float)w / (float)h, // aspect
            near, // near
            far // far
        );
        perspective[1][1] *= -1;
        
        glm::mat4 view = glm::lookAt(position, position + front, up);
        auto out = glm::vec4(coord, 1.f) * (perspective * view);
        out /= out.w;
        return glm::vec3(out);
    }
};