// === script.js ===
function runModel() {
    const toast = document.getElementById('toast');
    toast.innerText = 'Model Running... Please wait.';
    toast.style.opacity = '1';
    toast.style.bottom = '20px';
  
    setTimeout(() => {
      toast.innerText = 'Colorization Complete!';
      setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.bottom = '-50px';
      }, 3000);
    }, 2000); // Simulate model running delay
  }
  
  // Optional: Toast for upload
  document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
      const toast = document.getElementById('toast');
      toast.innerText = 'Image Uploaded Successfully!';
      toast.style.opacity = '1';
      toast.style.bottom = '20px';
  
      setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.bottom = '-50px';
      }, 3000);
    }
  });
  