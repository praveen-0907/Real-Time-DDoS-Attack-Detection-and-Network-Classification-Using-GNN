"""
check_dependencies.py - Verify GNNShield Dependencies
======================================================
Run this script to verify all required dependencies are installed correctly.

Usage:
    python check_dependencies.py
"""

import sys
import subprocess

def check_python_version():
    """Check Python version compatibility."""
    print("=" * 70)
    print("  Python Version Check")
    print("=" * 70)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✓ Python version is compatible (3.8-3.11)")
        return True
    else:
        print("✗ Python version not recommended. Use Python 3.8-3.11")
        return False

def check_package(package_name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and get its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, 'unknown')
        print(f"✓ {package_name:25s} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:25s} NOT INSTALLED")
        return False
    except Exception as e:
        print(f"⚠ {package_name:25s} ERROR: {e}")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ├─ CUDA available: YES")
            print(f"  ├─ CUDA version: {torch.version.cuda}")
            print(f"  ├─ GPU device: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  └─ VRAM: {vram:.1f} GB")
            return True
        else:
            print(f"  └─ CUDA available: NO (CPU only)")
            return False
    except Exception as e:
        print(f"  └─ CUDA check error: {e}")
        return False

def check_npcap():
    """Check if Npcap is installed (Windows only)."""
    if sys.platform != 'win32':
        return True
    
    try:
        result = subprocess.run(
            ['sc', 'query', 'npcap'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'RUNNING' in result.stdout:
            print("✓ Npcap service is running")
            return True
        else:
            print("✗ Npcap service not running")
            print("  Install from: https://npcap.com/#download")
            return False
    except Exception as e:
        print(f"⚠ Npcap check failed: {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print("  GNNShield Dependency Checker")
    print("=" * 70)
    print()
    
    results = {}
    
    # Check Python version
    results['python'] = check_python_version()
    print()
    
    # Check core dependencies
    print("=" * 70)
    print("  Core Dependencies")
    print("=" * 70)
    
    results['torch'] = check_package('PyTorch', 'torch')
    if results['torch']:
        results['cuda'] = check_torch_cuda()
    
    results['torch_geometric'] = check_package('PyTorch Geometric', 'torch_geometric')
    results['torch_scatter'] = check_package('torch-scatter', 'torch_scatter')
    results['torch_sparse'] = check_package('torch-sparse', 'torch_sparse')
    results['torch_cluster'] = check_package('torch-cluster', 'torch_cluster')
    results['torch_spline_conv'] = check_package('torch-spline-conv', 'torch_spline_conv')
    print()
    
    # Check data processing
    print("=" * 70)
    print("  Data Processing")
    print("=" * 70)
    results['numpy'] = check_package('NumPy', 'numpy')
    results['pandas'] = check_package('Pandas', 'pandas')
    print()
    
    # Check network tools
    print("=" * 70)
    print("  Network Tools")
    print("=" * 70)
    results['scapy'] = check_package('Scapy', 'scapy')
    if sys.platform == 'win32':
        results['npcap'] = check_npcap()
    print()
    
    # Check web framework
    print("=" * 70)
    print("  Web Framework")
    print("=" * 70)
    results['flask'] = check_package('Flask', 'flask')
    results['flask_socketio'] = check_package('Flask-SocketIO', 'flask_socketio')
    results['socketio'] = check_package('python-socketio', 'socketio')
    results['werkzeug'] = check_package('Werkzeug', 'werkzeug')
    print()
    
    # Check notifications
    print("=" * 70)
    print("  Notifications (Windows)")
    print("=" * 70)
    winotify_ok = check_package('winotify', 'winotify', version_attr='__version__')
    if not winotify_ok:
        win10toast_ok = check_package('win10toast', 'win10toast', version_attr='__version__')
        results['notifications'] = win10toast_ok
    else:
        results['notifications'] = True
    print()
    
    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    
    critical_deps = [
        'python', 'torch', 'torch_geometric', 'numpy', 'pandas', 
        'scapy', 'flask', 'flask_socketio'
    ]
    
    critical_ok = all(results.get(dep, False) for dep in critical_deps)
    optional_ok = results.get('cuda', False)
    
    if critical_ok:
        print("✓ All critical dependencies installed")
    else:
        print("✗ Some critical dependencies missing")
        missing = [dep for dep in critical_deps if not results.get(dep, False)]
        print(f"  Missing: {', '.join(missing)}")
    
    if optional_ok:
        print("✓ GPU support available (CUDA)")
    else:
        print("⚠ GPU support not available (CPU only)")
    
    if results.get('notifications', False):
        print("✓ Windows notifications available")
    else:
        print("⚠ Windows notifications not available")
    
    print()
    
    if critical_ok:
        print("=" * 70)
        print("  ✓ Ready to use GNNShield!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Download CICIDS dataset → data/cicids2017/")
        print("  2. Train model: python trainer.py --epochs 50")
        print("  3. Start detection: python detector.py")
        print("  4. Launch dashboard: python dashboard.py")
        print()
        return 0
    else:
        print("=" * 70)
        print("  ✗ Installation incomplete")
        print("=" * 70)
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("For detailed instructions, see INSTALLATION.md")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
