loading = false;
MAX_FILE_SIZE = 6553500

function handleDrop(event) {
    event.preventDefault();

    if (loading)
        return;

    loading = true;
    var files = event.dataTransfer.files;
    loadFile(files[0]);
}

function loadFile(file) {
    if (file.size > MAX_FILE_SIZE) {
        alert("File size is too big. Please upload a file less than 6.5MB.");
        loading = false;
        return;
    }

    document.querySelector('.loading').classList.add('is-visible');
    document.querySelector('#output').classList.add('is-loading');

    var output = document.getElementById('output');
    output.src = URL.createObjectURL(file);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/", true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            document.querySelector('.loading').classList.remove('is-visible');
            document.querySelector('#output').classList.remove('is-loading');

            var response = JSON.parse(xhr.responseText);
            var responseDiv = document.getElementById("response");
            responseDiv.classList.remove("success", "error");

            // var output = document.getElementById('output');
            // output.src = "data:image/png;base64," + response.heatmap;

            var percentage = (1 - response.ai_chance) * 100;
            document.querySelector('#percentage').innerHTML = percentage.toFixed(2) + '%' + " Human";
            document.querySelector('#progress').style.width = percentage + '%';
            if (percentage >= 50) {
                document.querySelector('#progress').style.backgroundColor = '#4CAF50';
            } else {
                document.querySelector('#progress').style.backgroundColor = '#F44336';
            }

            if (response.label === "human") {
                responseDiv.innerHTML = "HUMAN";
                responseDiv.classList.add("success");
            } else if (response.label === "ai") {
                responseDiv.innerHTML = "AI";
                responseDiv.classList.add("error");
            } else {
                responseDiv.innerHTML = response.label;
                responseDiv.classList.add("error");
            }
        }

        loading = false;
    };

    var formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
}