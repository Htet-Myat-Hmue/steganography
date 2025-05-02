function playSoundAndNavigate(url) {
    var audio = new Audio('/static/click.mp3');
    audio.play().catch(error => console.log("Audio play failed:", error));
    setTimeout(() => { window.location.href = url; }, 100);
}

function playSound() {
    var audio = new Audio('/static/click.mp3');
    audio.play().catch(error => console.log("Audio play failed:", error));
}

function showLoader() {
    document.getElementById('loader').style.display = 'block';
    document.querySelector('form').style.opacity = '0.5';
}

function validateFile(inputId, warningId) {
    var fileInput = document.getElementById(inputId);
    var warning = document.getElementById(warningId);
    if (!fileInput.files.length) {
        warning.style.display = 'block';
        return false; // Prevents form submission
    }
    warning.style.display = 'none';
    return true;
}

function validateEncoder(event) {
    var fileInput = document.getElementById('cover_image');
    var warning = document.getElementById('cover_warning');
    if (!fileInput.files.length) {
        warning.style.display = 'block';
        event.preventDefault(); // Explicitly stop form submission
        return false;
    }
    showLoader();
    return true;
}

function validateDecoder(event) {
    var fileInput = document.getElementById('stego_image');
    var warning = document.getElementById('stego_warning');
    var fileName = document.getElementById('stego_name').textContent;
    if (!fileInput.files.length && !fileName) {
        warning.style.display = 'block';
        event.preventDefault();
        return false;
    }
    showLoader();
    return true;
}

function hideWarning(warningId) {
    document.getElementById(warningId).style.display = 'none';
}

function toggleManual() {
    var method = document.getElementById('method').value;
    document.getElementById('manual_options').style.display = method === 'manual' ? 'block' : 'none';
}

function copyToClipboard(text, notiId) {
    navigator.clipboard.writeText(text).then(() => {
        var noti = document.getElementById(notiId);
        noti.style.display = 'inline';
        setTimeout(() => { noti.style.display = 'none'; }, 2000);
    });
}

function showFileName(inputId, displayId) {
    var fileInput = document.getElementById(inputId);
    var display = document.getElementById(displayId);
    if (fileInput.files.length > 0) {
        display.textContent = "Selected: " + fileInput.files[0].name;
    } else {
        display.textContent = "";
    }
}