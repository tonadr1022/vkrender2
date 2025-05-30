#version 460

// https://github.com/vblanco20-1/vulkan-guide/blob/engine/shaders/fullscreen.vert
layout(location = 0) out vec2 out_uv;

void main() {
    out_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(out_uv * 2.0f - 1.0f, 0.1f, 1.0f);
}
