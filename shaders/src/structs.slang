module structs;

public enum ObjectType {
	Triangle,
	Sphere
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

	public __init() {
		radiance = float3(0.0);
		attenuation = float3(1.0);
		done = false;
		origin = float3(0.0);
		direction = float3(0.0);
		is_light = false;
	}
};

public struct Material {
	public float3 albedo;
	public float alpha;
	public float refraction_index;
	public float smoothness;
	public float emissivity;
	public float emission_strength;
	public float3 emission_colour;
	public uint padding;

	public __init() {
		albedo = float3(0.5, 0.5, 0.5);
		alpha = 1.0;
		refraction_index = 1.0;
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

public struct Mesh {
	public uint object_type;
	public Vertex* vertices;
	public uint* indices;
	public uint* material_indices;
}

public struct Sphere {
	public uint object_type;
	public float3 center;
	public float radius;
	public uint material;
	public uint is_selected;
}

public struct Buffers {
	public uint64_t primitive;
}

public struct CameraUniforms {
	public float3 origin;
	public float focal_distance;
	public float3 u;
	public float vfov;
	public float3 v;
	public float dof_scale;
	public float3 w;
}

public struct UBO {
	public CameraUniforms camera;
	public uint frame_num;
	public uint ray_frame_num;
	public uint width;
	public uint height;
}
