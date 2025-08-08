module structs;

public enum ObjectType {
	Triangle,
	Sphere,
	Ocean,
	Lentil,
}

public struct CustomAttributes {
	public uint hit_type;
}

public struct Attributes {
	public float2 bary;
};

public struct Payload {
	[[vk::location(0)]]
	public float3 radiance;
	public float3 attenuation;
	public int done;
	public float3 origin;
	public float3 direction;
	public bool is_light;
	public float refractive_index;

	public __init() {
		radiance = float3(0.0);
		attenuation = float3(1.0);
		done = false;
		origin = float3(0.0);
		direction = float3(0.0);
		is_light = false;
		refractive_index = 1.0;
	}
};

public struct Material {
	public float3 albedo;
	public float alpha;
	public float refractive_index;
	public float smoothness;
	public float emissivity;
	public float emission_strength;
	public float3 emission_colour;
	public uint padding;

	public __init() {
		albedo = float3(0.5, 0.5, 0.5);
		alpha = 1.0;
		refractive_index = 1.0;
		smoothness = 0.0;
		emissivity = 0.0;
		emission_strength = 0.0;
		emission_colour = float3(1.0, 1.0, 1.0);
		padding = 0;
	}
}

public struct Vertex {
	public float3 pos;
	uint pad1;
	public float3 normal;
	uint pad2;
}

public struct AABB {
	public float3 min;
	public float3 max;
	public __init(float3 minimum, float3 maximum) {
		min = minimum;
		max = maximum;
	}
}

public struct Mesh {
	public uint64_t object_type;
	public Vertex* vertices;
	public uint* indices;
	public uint* material_indices;
}

public struct Sphere {
	public uint64_t object_type;
	public float radius;
	public uint material;
	public uint is_selected;
}

public struct Ocean {
	public uint64_t object_type;
	public uint material;
	public uint size;
}

public struct Lentil {
	public uint64_t object_type;
	public float radius;
	public float length;
	public float curvature_a;
	public float curvature_b;
	public float kappa_a;
	public float kappa_b;
	public uint material;
	public uint is_selected;
}

public struct Buffers {
	public uint64_t primitive;
}

public struct Camera {
	public float3 origin;
	public float vfov;
	public float4 q;
	public float focal_distance;
	public float dof_scale;
	public uint enabled;
	public uint padding;
}

public struct UBO {
	public Camera camera;
	public uint ray_frame_num;
	public uint width;
	public uint height;
	public uint random_num;
}

public struct RayPushConstants {
	public RaytracingAccelerationStructure* tlas;  // Pointer to RaytracingAccelerationStructure
	public UBO* ubo;							   // Pointer to UBO (should switch every time)
	public uint64_t scene;						   // Pointer to array of Buffers
	public uint64_t materials;					   // Pointer to array of Material
	public uint32_t frame_index;
}
