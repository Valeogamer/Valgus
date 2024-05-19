document.getElementById('fileInput').onchange = function(e) {
    var reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('imagePreview').src = event.target.result;
    }
    reader.readAsDataURL(this.files[0]);
};
