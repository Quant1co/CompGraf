#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

/* =======================
TEXTURE
======================= */
uniform sampler2D diffuseTex;

/* =======================
CAMERA
======================= */
uniform vec3 viewPos;

/* =======================
LIGHT STRUCTS
======================= */

// 1) Directional (Global)
struct DirLight {
vec3 direction;
vec3 color;
};

// 2) Point light
struct PointLight {
vec3 position;
vec3 color;
float constant;
float linear;
float quadratic;
};

// 3) Spot light
struct SpotLight {
vec3 position;
vec3 direction;
vec3 color;
float cutOff;
float outerCutOff;
};

/* =======================
LIGHT UNIFORMS
======================= */
uniform DirLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;

/* =======================
SHADING MODEL SELECT
0 - Phong
1 - Toon
2 - Oren–Nayar
======================= */
uniform int shadingModel;

/* =======================
FUNCTIONS
======================= */

// -------- Phong --------
vec3 phong(vec3 lightDir, vec3 lightColor)
{
vec3 N = normalize(Normal);
vec3 L = normalize(lightDir);
vec3 V = normalize(viewPos - FragPos);
vec3 R = reflect(-L, N);
float diff = max(dot(N, L), 0.0);
float spec = pow(max(dot(V, R), 0.0), 32.0);

vec3 ambient = 0.1 * lightColor;
return ambient + diff * lightColor + spec * lightColor;
}

// -------- Toon --------
vec3 toon(vec3 lightDir, vec3 lightColor)
{
vec3 N = normalize(Normal);
vec3 L = normalize(lightDir);
float d = dot(N, L);

float level;
if (d > 0.8) level = 1.0;
else if (d > 0.5) level = 0.7;
else if (d > 0.25) level = 0.4;
else level = 0.1;

return lightColor * level;
}

// -------- Oren–Nayar --------
vec3 orenNayar(vec3 lightDir, vec3 lightColor)
{
vec3 N = normalize(Normal);
vec3 L = normalize(lightDir);
vec3 V = normalize(viewPos - FragPos);
float roughness = 0.6;
float sigma2 = roughness * roughness;

float NL = max(dot(N, L), 0.0);
float NV = max(dot(N, V), 0.0);

float A = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
float B = 0.45 * sigma2 / (sigma2 + 0.09);

vec3 Lproj = normalize(L - N * NL);
vec3 Vproj = normalize(V - N * NV);

float gamma = max(dot(Lproj, Vproj), 0.0);

float diff = NL * (A + B * gamma * sqrt((1.0 - NL*NL)*(1.0 - NV*NV)));

return lightColor * diff;
}

// -------- Select model --------
vec3 applyModel(vec3 lightDir, vec3 lightColor)
{
if (shadingModel == 0)
return phong(lightDir, lightColor);
else if (shadingModel == 1)
return toon(lightDir, lightColor);
else
return orenNayar(lightDir, lightColor);
}

/* =======================
MAIN
======================= */
void main()
{
vec3 result = vec3(0.0);

// -------- Directional --------
result += applyModel(-dirLight.direction, dirLight.color);

// -------- Point --------
float dist = length(pointLight.position - FragPos);
float attenuation = 1.0 / (
    pointLight.constant +
    pointLight.linear * dist +
    pointLight.quadratic * dist * dist
);

result += applyModel(pointLight.position - FragPos, pointLight.color)
          * attenuation;

// -------- Spot --------
vec3 L = normalize(spotLight.position - FragPos);
float theta = dot(L, normalize(-spotLight.direction));
float epsilon = spotLight.cutOff - spotLight.outerCutOff;
float intensity = clamp((theta - spotLight.outerCutOff) / epsilon, 0.0, 1.0);

result += applyModel(spotLight.position - FragPos, spotLight.color)
          * intensity;

// -------- Texture --------
vec3 texColor = texture(diffuseTex, TexCoord).rgb;
FragColor = vec4(result * texColor, 1.0);
}