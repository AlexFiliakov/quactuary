"""Verify Qiskit installation for quActuary quantum module."""

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    print("✓ Qiskit core imports successful")
    
    # Test basic circuit creation
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    print("✓ Basic circuit creation successful")
    
    # Test simulation
    backend = AerSimulator()
    job = backend.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()
    print(f"✓ Bell state measurements: {counts}")
    
    # Verify algorithm imports
    from qiskit_algorithms import (
        AmplitudeEstimation, 
        MaximumLikelihoodAmplitudeEstimation,
        IterativeAmplitudeEstimation,
        EstimationProblem
    )
    print("✓ Qiskit algorithms imports successful")
    
    print("\n✓ All Qiskit components verified successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error during verification: {e}")