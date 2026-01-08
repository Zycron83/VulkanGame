// Vertex shader parameters
struct VertexInput
{
    float3 position : POSITION;
    float3 color : COLOR;
    float3 normal;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR;
    float3 normal : NORMAL;
    float2 point : CORNER;
};

// Uniform buffer
struct UniformBuffer
{
    float time;
    float4x4 mvp;
    float3x3 normalMatrix;
    float2 point;
};
[[vk_binding(0, 0)]] ConstantBuffer<UniformBuffer, ScalarDataLayout> ubo;

// Vertex shader
[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uint index : SV_VertexID)
{
    VertexOutput output;
    output.position = mul(ubo.mvp, float4(input.position, 1.0));

    output.normal = mul(ubo.normalMatrix, input.normal);
    uint cube_index = index / 24;
    output.color = abs(normalize(output.normal));

    output.point = ubo.point;
    return output;
}

// Fragment shader
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    if (distance(input.position.xy, input.point) < 10) {
        return float4(1);
    }
    // let up = float3(0, 1, 0);
    // let color = lerp(float3(1,1,1), float3(1,0,0), max(0.0, dot(up, input.normal)));
    return (float4(input.color, 1.0));
}