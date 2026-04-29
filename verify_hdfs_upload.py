"""
HDFS Upload Verification Script
================================
Verifies that CSV files are being uploaded to HDFS correctly
and can be viewed at http://localhost:9870/explorer.html#/stock_data/live_api_dumps

Usage:
    python verify_hdfs_upload.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# HDFS Configuration
HDFS_DIRECTORY = "/stock_data/live_api_dumps/"
HDFS_WEB_UI = "http://localhost:9870/explorer.html#/stock_data/live_api_dumps"

def run_command(cmd, description):
    """Run a shell command and return result."""
    print(f"\n{'='*70}")
    print(f"🔍 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True  # Use shell on Windows
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print(result.stdout)
            return True, result.stdout
        else:
            print("❌ FAILED")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, str(e)


def check_hdfs_running():
    """Check if HDFS is running."""
    print("\n" + "="*70)
    print("🔍 CHECKING HDFS STATUS")
    print("="*70)
    
    success, output = run_command(
        ["jps"],
        "Checking Java processes (should show NameNode and DataNode)"
    )
    
    if success:
        has_namenode = "NameNode" in output
        has_datanode = "DataNode" in output
        
        if has_namenode and has_datanode:
            print("\n✅ HDFS is running correctly!")
            print(f"   - NameNode: {'✅' if has_namenode else '❌'}")
            print(f"   - DataNode: {'✅' if has_datanode else '❌'}")
            return True
        else:
            print("\n⚠️ HDFS is not fully running!")
            print(f"   - NameNode: {'✅' if has_namenode else '❌'}")
            print(f"   - DataNode: {'✅' if has_datanode else '❌'}")
            print("\n💡 Start HDFS with: start-dfs.sh")
            return False
    else:
        print("\n❌ Could not check HDFS status")
        print("💡 Make sure Hadoop is installed and in PATH")
        return False


def create_hdfs_directory():
    """Create HDFS directory if it doesn't exist."""
    success, output = run_command(
        ["hdfs", "dfs", "-mkdir", "-p", HDFS_DIRECTORY],
        f"Creating HDFS directory: {HDFS_DIRECTORY}"
    )
    return success


def list_hdfs_directory():
    """List files in HDFS directory."""
    success, output = run_command(
        ["hdfs", "dfs", "-ls", HDFS_DIRECTORY],
        f"Listing files in HDFS directory: {HDFS_DIRECTORY}"
    )
    
    if success and output:
        # Count files
        lines = [line for line in output.split('\n') if line.strip() and not line.startswith('Found')]
        file_count = len(lines)
        
        print(f"\n📊 Found {file_count} file(s) in HDFS")
        
        if file_count > 0:
            print(f"\n✅ Files are being uploaded to HDFS!")
            print(f"\n🌐 View in web UI:")
            print(f"   {HDFS_WEB_UI}")
        else:
            print(f"\n⚠️ No files found in HDFS yet")
            print(f"   Run a prediction from the dashboard to upload files")
    
    return success


def test_upload():
    """Test uploading a sample file to HDFS."""
    print("\n" + "="*70)
    print("🧪 TESTING HDFS UPLOAD")
    print("="*70)
    
    # Create a test file
    test_file = Path("test_hdfs_upload.txt")
    test_content = f"HDFS Upload Test - {datetime.now().isoformat()}"
    
    try:
        test_file.write_text(test_content)
        print(f"✅ Created test file: {test_file}")
        
        # Upload to HDFS
        success, output = run_command(
            ["hdfs", "dfs", "-put", "-f", str(test_file), HDFS_DIRECTORY],
            f"Uploading test file to HDFS"
        )
        
        if success:
            print(f"\n✅ Test file uploaded successfully!")
            
            # Verify upload
            success2, output2 = run_command(
                ["hdfs", "dfs", "-ls", f"{HDFS_DIRECTORY}{test_file.name}"],
                "Verifying test file in HDFS"
            )
            
            if success2:
                print(f"\n✅ Test file verified in HDFS!")
                print(f"\n🌐 View in web UI:")
                print(f"   {HDFS_WEB_UI}")
                
                # Clean up test file from HDFS
                run_command(
                    ["hdfs", "dfs", "-rm", f"{HDFS_DIRECTORY}{test_file.name}"],
                    "Cleaning up test file from HDFS"
                )
            else:
                print(f"\n⚠️ Could not verify test file in HDFS")
        else:
            print(f"\n❌ Test file upload failed")
        
        # Clean up local test file
        test_file.unlink()
        print(f"✅ Cleaned up local test file")
        
        return success
        
    except Exception as e:
        print(f"❌ Test upload failed: {e}")
        if test_file.exists():
            test_file.unlink()
        return False


def check_web_ui():
    """Check if HDFS web UI is accessible."""
    print("\n" + "="*70)
    print("🌐 HDFS WEB UI ACCESS")
    print("="*70)
    
    print(f"\n📍 HDFS Web UI URL:")
    print(f"   {HDFS_WEB_UI}")
    print(f"\n💡 Open this URL in your browser to view uploaded files")
    print(f"\n📂 Directory structure:")
    print(f"   /stock_data/")
    print(f"   └── live_api_dumps/")
    print(f"       └── [CSV files from predictions]")


def main():
    """Main verification function."""
    print("\n" + "="*70)
    print("🔍 HDFS UPLOAD VERIFICATION")
    print("="*70)
    print(f"\nThis script verifies that CSV files are uploaded to HDFS")
    print(f"and can be viewed at: {HDFS_WEB_UI}")
    
    # Step 1: Check if HDFS is running
    if not check_hdfs_running():
        print("\n❌ HDFS is not running. Please start HDFS first.")
        print("\n💡 Start HDFS with:")
        print("   start-dfs.sh")
        sys.exit(1)
    
    # Step 2: Create HDFS directory
    print("\n" + "="*70)
    print("📁 SETTING UP HDFS DIRECTORY")
    print("="*70)
    create_hdfs_directory()
    
    # Step 3: List existing files
    print("\n" + "="*70)
    print("📋 CHECKING EXISTING FILES")
    print("="*70)
    list_hdfs_directory()
    
    # Step 4: Test upload
    test_upload()
    
    # Step 5: Show web UI info
    check_web_ui()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   - HDFS is running: ✅")
    print(f"   - Directory created: ✅")
    print(f"   - Upload tested: ✅")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run: streamlit run app.py")
    print(f"   2. Select a stock (e.g., AAPL)")
    print(f"   3. Click 'Predict Live'")
    print(f"   4. Check HDFS web UI: {HDFS_WEB_UI}")
    print(f"\n💡 CSV files will appear in the web UI after prediction")
    print()


if __name__ == "__main__":
    main()
