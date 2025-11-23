import subprocess
import os
import sys
import subprocess
import platform

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    if os.path.exists("requirements.txt"):
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print("❌ Failed to install requirements!")
            print("STDERR:", result.stderr)
            return False
    else:
        print("❌ requirements.txt not found!")
        return False
    

#main TODO:
#    uisti sa ze ma vsetky requirements.txt
#
#    ak neexistuje best_1.pt to iste co dole nad best_2.pt ale s datasetom 1.
#
#    ak neexistuje best_2.pt, 
#        stiahne best_2.pt (gdown). 
#        ak sa neda stiahnut, 
#            pusti trenovanie nad dataset 2
#            ak neexistuje dataset 2, 
#                stiahne dataset 2 (gdown).
#                ak sa neda stiahnut, error.
#                rozbali a presunie do prislusnej slozky.
#
#    ak neexistuje best_3.pt, pusti trenovanie nad oboma datasetmi
#
#
#    otestuje best_1.pt, best_2.pt, best_3.pt nad oboma datasetmi, a urobi grafy presnosti a podobne porovnania.
#
#
#    pomocou cv2? tvori 3x3 oknove video, kde budu vsetky 3 modely trackovat na testovacom videu, s vyuzitim oboch trackerov. a posledne 3 okna budu nejake zmysluplne statistiky v case, ktore budu ukazovat rozdiely v modeloch a trackeroch. napr. heatmapy ktore modely a trackeri sa kedy zhodli a nezhodli?
#    toto sa pocita paralelne, a zobrazuje sa a uklada sa do output videa.


def main():
    """Main build and run function"""
    print("🚀 PGR tracker - Build and Run")
    print("=" * 50)
    
    # Check platform
    system = platform.system()
    print(f"💻 Platform: {system} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Install requirements first
    if not install_requirements():
        print("⚠️  Continuing with build...")
    
    import gdown


    folder_id = "1zfFXMRVMnHY0xS7eI3EbHi3oTY3fBAH_"
    model1_id = "best.pt"
    dataset1 = "Citypersons.v1i.yolov8.zip"
    url = f"https://drive.google.com/drive/folders/{folder_id}/{model1_id}" # Construct the URL for downloading the file

    output_dir = "models/1"

    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists
    output_path = os.path.join(output_dir, "best.pt")

    # Download the file
    gdown.download(url, output=output_path, quiet=False)

    print(f"The file {model1_id} has been downloaded to: {output_path}")

if __name__ == "__main__":
    main()