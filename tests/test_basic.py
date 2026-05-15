import numpy as np

def test_import():
    """Test that the module can be imported"""
    try:
        import cufloda
        print("✓ Module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_lbm_creation():
    """Test creating an LBM simulation"""
    try:
        import cufloda
        sim = cufloda.LBM2D(64, 64, 0.02)
        print(f"✓ LBM2D created: {sim.width}x{sim.height}")
        return True
    except Exception as e:
        print(f"✗ LBM2D creation failed: {e}")
        return False

def test_initialization():
    """Test simulation initialization"""
    try:
        import cufloda
        sim = cufloda.LBM2D(64, 64, 0.02)
        sim.initialize(1.0, 0.1, 0.0)
        density = sim.get_density()
        print(f"✓ Initialization successful, density shape: {density.shape}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_step():
    """Test single simulation step"""
    try:
        import cufloda
        sim = cufloda.LBM2D(64, 64, 0.02)
        sim.initialize(1.0, 0.1, 0.0)
        sim.step()
        print("✓ Step executed successfully")
        return True
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False

def test_obstacles():
    """Test obstacle setting"""
    try:
        import cufloda
        sim = cufloda.LBM2D(64, 64, 0.02)
        obstacles = np.zeros((64, 64), dtype=bool)
        obstacles[32:40, 32:40] = True
        sim.set_obstacles(obstacles)
        print("✓ Obstacles set successfully")
        return True
    except Exception as e:
        print(f"✗ Obstacle setting failed: {e}")
        return False

if __name__ == "__main__":
    print("Running CuFloda tests...")
    print("-" * 40)
    
    tests = [
        test_import,
        test_lbm_creation,
        test_initialization,
        test_step,
        test_obstacles,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("-" * 40)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
