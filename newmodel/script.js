async function colorizeUploadedImage() {
  const fileInput = document.getElementById('fileInput');
  const outputImage = document.getElementById('outputImage');  // Correct ID
  const file = fileInput.files[0];

  if (!file) {
    alert('Please select a file first!');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://127.0.0.1:5000/colorize', {  // Correct endpoint
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      outputImage.src = url;  // Correct output image
      document.getElementById('outputImageContainer').style.display = 'flex';// <-- SHOW
      outputImage.style.display = 'block';

      showToast('Image Colorized Successfully!');
    } else {
      alert('Failed to colorize image!');
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error uploading file!');
  }
}

// Toast notification
function showToast(message) {
  const toast = document.getElementById('toast');
  if (toast) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3000);
  }
}

// Fade-in effect
window.addEventListener('load', function () {
  document.body.classList.add('loaded');
});
// Show Upload Section when clicking Try Our Demo
function showUpload() {
  const uploadSection = document.getElementById('upload-section');
  uploadSection.style.display = 'flex';
  uploadSection.scrollIntoView({ behavior: 'smooth' });
}
