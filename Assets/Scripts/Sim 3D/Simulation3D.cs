using Unity.Mathematics;
using UnityEngine;

public class Simulation3D : MonoBehaviour
{
    public event System.Action SimulationStepCompleted;

    [Header("Settings")]
    public float timeScale = 1;
    public bool fixedTimeStep;
    public int iterationsPerFrame;
    public float gravity = -10;

    [Range(0, 1)]
    public float collisionDamping = 0.05f;
    public float smoothingRadius = 0.2f;
    public float targetDensity;
    public float pressureMultiplier;
    public float nearPressureMultiplier;
    public float viscosityStrength;

    [Header("References")]
    public ComputeShader compute;
    public Spawner3D spawner;
    public ParticleDisplay3D display;
    public Transform floorDisplay;

    // Buffers
    public ComputeBuffer positionBuffer { get; private set; }
    public ComputeBuffer velocityBuffer { get; private set; }
    public ComputeBuffer densityBuffer { get; private set; }
    public ComputeBuffer predictedPositionsBuffer;
    ComputeBuffer spatialIndices;
    ComputeBuffer spatialOffsets;

    // Kernel IDs
    const int externalForcesKernel = 0;
    const int spatialHashKernel = 1;
    const int densityKernel = 2;
    const int pressureKernel = 3;
    const int viscosityKernel = 4;
    const int updatePositionsKernel = 5;

    GPUSort gpuSort;

    // State
    bool isPaused;
    bool pauseNextFrame;
    Spawner3D.SpawnData spawnData;

    [Header("Interactive Objects")]
    public Transform interactiveSphere;
    private Vector3 sphereVelocity;
    private ComputeBuffer sphereVelocityBuffer;

    void Start()
    {
        Debug.Log("Controls: Space = Play/Pause, R = Reset");
        Debug.Log("Use transform tool in scene to scale/rotate simulation bounding box.");

        float deltaTime = 1 / 60f;
        Time.fixedDeltaTime = deltaTime;

        spawnData = spawner.GetSpawnData();

        // Create buffers
        int numParticles = spawnData.points.Length;
        positionBuffer = ComputeHelper.CreateStructuredBuffer<float3>(numParticles);
        predictedPositionsBuffer = ComputeHelper.CreateStructuredBuffer<float3>(numParticles);
        velocityBuffer = ComputeHelper.CreateStructuredBuffer<float3>(numParticles);
        densityBuffer = ComputeHelper.CreateStructuredBuffer<float2>(numParticles);
        spatialIndices = ComputeHelper.CreateStructuredBuffer<uint3>(numParticles);
        spatialOffsets = ComputeHelper.CreateStructuredBuffer<uint>(numParticles);

        // Set buffer data
        SetInitialBufferData(spawnData);

        // Init compute
        ComputeHelper.SetBuffer(
            compute,
            positionBuffer,
            "Positions",
            externalForcesKernel,
            updatePositionsKernel
        );
        ComputeHelper.SetBuffer(
            compute,
            predictedPositionsBuffer,
            "PredictedPositions",
            externalForcesKernel,
            spatialHashKernel,
            densityKernel,
            pressureKernel,
            viscosityKernel,
            updatePositionsKernel
        );
        ComputeHelper.SetBuffer(
            compute,
            spatialIndices,
            "SpatialIndices",
            spatialHashKernel,
            densityKernel,
            pressureKernel,
            viscosityKernel
        );
        ComputeHelper.SetBuffer(
            compute,
            spatialOffsets,
            "SpatialOffsets",
            spatialHashKernel,
            densityKernel,
            pressureKernel,
            viscosityKernel
        );
        ComputeHelper.SetBuffer(
            compute,
            densityBuffer,
            "Densities",
            densityKernel,
            pressureKernel,
            viscosityKernel
        );
        ComputeHelper.SetBuffer(
            compute,
            velocityBuffer,
            "Velocities",
            externalForcesKernel,
            pressureKernel,
            viscosityKernel,
            updatePositionsKernel
        );

        // Initialize the sphere velocity buffer
        sphereVelocityBuffer = new ComputeBuffer(1, sizeof(float) * 3);
        sphereVelocityBuffer.SetData(new[] { sphereVelocity });
        compute.SetBuffer(updatePositionsKernel, "SphereVelocity", sphereVelocityBuffer);

        compute.SetInt("numParticles", positionBuffer.count);

        gpuSort = new();
        gpuSort.SetBuffers(spatialIndices, spatialOffsets);

        // Init display
        display.Init(this);
    }

    void FixedUpdate()
    {
        // Run simulation if in fixed timestep mode
        if (fixedTimeStep)
        {
            RunSimulationFrame(Time.fixedDeltaTime);
        }
    }

    void Update()
    {
        if (!fixedTimeStep && Time.frameCount > 10)
        {
            RunSimulationFrame(Time.deltaTime);
        }

        if (pauseNextFrame)
        {
            isPaused = true;
            pauseNextFrame = false;
        }

        floorDisplay.transform.localScale = new Vector3(1, 1 / transform.localScale.y * 0.1f, 1);

        if (interactiveSphere != null)
        {
            sphereVelocity += Physics.gravity * Time.deltaTime;
            Vector3 newPosition = interactiveSphere.position + sphereVelocity * Time.deltaTime;

            // Check for collisions
            ResolveSphereCollisionInUnity(ref newPosition, ref sphereVelocity);

            interactiveSphere.position = newPosition; // update position after collision resolution

            // Set the updated sphere's position and velocity in the compute shader
            compute.SetVector(
                "spherePosition",
                new Vector4(newPosition.x, newPosition.y, newPosition.z, 0)
            );
            compute.SetFloat("sphereRadius", interactiveSphere.localScale.x * 0.5f);
            // Debug the magnitude of the sphere's velocity
            sphereVelocityBuffer.SetData(new[] { sphereVelocity });
            // Set buffer for all kernels that use it
            compute.SetBuffer(updatePositionsKernel, "SphereVelocity", sphereVelocityBuffer);
            Debug.Log("Sphere Velocity: " + sphereVelocity.magnitude);
        }

        HandleInput();
    }

    void ResolveSphereCollisionInUnity(ref Vector3 position, ref Vector3 velocity)
    {
        float radius = interactiveSphere.localScale.x * 0.5f;

        // Assuming the floor is a horizontal plane at y = 0 or at its own y position
        float floorY =
            floorDisplay.transform.position.y + (floorDisplay.transform.localScale.y * 0.5f);

        // Check for collision with the floor
        if (position.y - radius < floorY)
        {
            position.y = floorY + radius;
            velocity.y = -velocity.y * collisionDamping * 0.5f;
        }
        // when the velocity is very small, stop the sphere from moving
        if (Mathf.Abs(velocity.y) < 0.01f)
        {
            velocity.y = 0;
        }
    }

    void RunSimulationFrame(float frameTime)
    {
        if (!isPaused)
        {
            float timeStep = frameTime / iterationsPerFrame * timeScale;

            UpdateSettings(timeStep);

            for (int i = 0; i < iterationsPerFrame; i++)
            {
                RunSimulationStep();
                SimulationStepCompleted?.Invoke();
            }
        }
    }

    void RunSimulationStep()
    {
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: externalForcesKernel);
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: spatialHashKernel);
        gpuSort.SortAndCalculateOffsets();
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: densityKernel);
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: pressureKernel);
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: viscosityKernel);
        ComputeHelper.Dispatch(compute, positionBuffer.count, kernelIndex: updatePositionsKernel);
    }

    void UpdateSettings(float deltaTime)
    {
        Vector3 simBoundsSize = transform.localScale;
        Vector3 simBoundsCentre = transform.position;

        compute.SetFloat("deltaTime", deltaTime);
        compute.SetFloat("gravity", gravity);
        compute.SetFloat("collisionDamping", collisionDamping);
        compute.SetFloat("smoothingRadius", smoothingRadius);
        compute.SetFloat("targetDensity", targetDensity);
        compute.SetFloat("pressureMultiplier", pressureMultiplier);
        compute.SetFloat("nearPressureMultiplier", nearPressureMultiplier);
        compute.SetFloat("viscosityStrength", viscosityStrength);
        compute.SetVector("boundsSize", simBoundsSize);
        compute.SetVector("centre", simBoundsCentre);

        compute.SetMatrix("localToWorld", transform.localToWorldMatrix);
        compute.SetMatrix("worldToLocal", transform.worldToLocalMatrix);
    }

    void SetInitialBufferData(Spawner3D.SpawnData spawnData)
    {
        float3[] allPoints = new float3[spawnData.points.Length];
        System.Array.Copy(spawnData.points, allPoints, spawnData.points.Length);

        positionBuffer.SetData(allPoints);
        predictedPositionsBuffer.SetData(allPoints);
        velocityBuffer.SetData(spawnData.velocities);
    }

    void HandleInput()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isPaused = !isPaused;
        }

        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            isPaused = false;
            pauseNextFrame = true;
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            isPaused = true;
            SetInitialBufferData(spawnData);
        }
    }

    void OnDestroy()
    {
        ComputeHelper.Release(
            positionBuffer,
            predictedPositionsBuffer,
            velocityBuffer,
            densityBuffer,
            spatialIndices,
            spatialOffsets,
            sphereVelocityBuffer
        );
    }

    void OnDrawGizmos()
    {
        // Draw Bounds
        var m = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.color = new Color(0, 1, 0, 0.5f);
        Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
        Gizmos.matrix = m;
    }
}
