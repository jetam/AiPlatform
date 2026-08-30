import { apiFetch } from "./api.js";

let dropzone = document.getElementById("dropzone");
let modelSelect = document.getElementById("modelSelect");
let generateButton = document.getElementById("generate");
let downloadButton = document.getElementById("downloadMidi");

// ordered top-to-bottom: hiding a step should hide everything after it too
const steps = [modelSelect, generateButton, downloadButton];

function hideFrom(step) {
    const idx = steps.indexOf(step);
    for (let i = idx; i < steps.length; i++) {
        steps[i].classList.add("d-none");
    }
}

let currentFileName = "";
const defaultDropzoneText = dropzone.textContent;

function showError(message) {
    console.error(message);
    alert(message);
}

function onGenerated(blob) {
    downloadButton.classList.remove("d-none");
    downloadButton.onclick = () => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "generated.mid";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
}

dropzone.addEventListener("dragover", (e) => {
    dropzone.style.background = "purple";
    e.preventDefault();
});

async function uploadMIDI( midiFile ) {
    const formData = new FormData();
    formData.append("midiFile", midiFile);

    let response = await apiFetch("/api/music/midi/upload", {
        method: "POST",
        body: formData
    });

    if ( !response.ok ) {
        throw new Error(`Upload failed (${response.status})`);
    }

    const res = await response.json();
        console.log( "uploadMIDI: result:" );
        console.log( res );
}

async function fetchModels() {
    let response = await apiFetch("/api/music/model/selection");

    if ( !response.ok ) {
        throw new Error(`Failed to load models (${response.status})`);
    }

    const res = await response.json();

    modelSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select model";
    placeholder.selected = true;
    modelSelect.appendChild(placeholder);

    for (const model of res.models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    }
}

generateButton.addEventListener("click", async (e) => {

    dropzone.innerHTML = `
        <div class="spinner-border text-light" role="status"></div>
        <span class="ms-2">generating MIDI...</span>
    `;

    generateButton.classList.add("d-none");


    e.preventDefault();

    hideFrom(downloadButton);

    try {
        const response = await apiFetch("/api/music/generate", {
            method: "POST"
        });

        const contentType = response.headers.get("content-type") || "";

        if ( contentType.includes("application/json") ) {
            const res = await response.json();
            if ( res.status === "Model Not Selected" ) {
                showError( "Model is not selected!" );
            } else {
                showError( `Generation failed: ${res.detail || res.status || "unknown error"}` );
            }
            return;
        }

        if ( !response.ok ) {
            throw new Error(`Generation failed (${response.status})`);
        }

        const blob = await response.blob();
        onGenerated( blob );
        dropzone.textContent = currentFileName;
    } catch (err) {
        showError( "Failed to generate music: " + err.message );
        dropzone.textContent = currentFileName;
    } finally {
        generateButton.classList.remove("d-none");
    }
});

modelSelect.addEventListener("change", async (e) => {
    hideFrom(generateButton);

    dropzone.innerHTML = `
        <div class="spinner-border text-light" role="status"></div>
        <span class="ms-2">fine tuning...</span>
    `;

    try {
        const response = await apiFetch("/api/music/model/select", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: e.target.value })
        });

        if ( !response.ok ) {
            throw new Error(`Model selection failed (${response.status})`);
        }

        const placeholder = modelSelect.querySelector('option[value=""]');
        if (placeholder) placeholder.remove();

        generateButton.classList.remove("d-none");
        dropzone.textContent = currentFileName;
    } catch (err) {
        showError( "Failed to select/fine-tune model: " + err.message );
        modelSelect.value = "";
        dropzone.textContent = currentFileName;
    }
});

dropzone.addEventListener("drop", async (e) => {

    e.preventDefault();


    const file = e.dataTransfer.files[0];

    console.log( "file dropped. File name = " + file.name );

    const formData = new FormData();
    formData.append("file", file);

    hideFrom(modelSelect);

    dropzone.innerHTML = `
        <div class="spinner-border text-light" role="status"></div>
        <span class="ms-2">fine tuning...</span>
    `;

    try {
        await uploadMIDI( file );
        await fetchModels();
        modelSelect.classList.remove("d-none");
        currentFileName = file.name;
        dropzone.textContent = currentFileName;
    } catch (err) {
        showError( "Failed to upload/process the file: " + err.message );
        currentFileName = "";
        dropzone.textContent = defaultDropzoneText;
    }
});
