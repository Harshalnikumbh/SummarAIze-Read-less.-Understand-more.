document.getElementById("summarizeBtn").addEventListener("click", async () => {
  const url = document.getElementById("urlInput").value.trim();
  const summaryBox = document.getElementById("summaryBox");
  const btn = document.getElementById("summarizeBtn");
  
  if (!url) {
    alert("Please paste a URL first!");
    return;
  }

  // Show loading state
  btn.disabled = true;
  btn.textContent = "Summarizing...";
  summaryBox.textContent = "";
  summaryBox.style.display = "none";

  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    const data = await response.json();
    
    // Display only the summary
    if (data.summary) {
      summaryBox.textContent = data.summary;
      summaryBox.classList.add("show");
    } else {
      summaryBox.textContent = "No summary could be generated for this URL.";
    }
    
  } catch (err) {
    console.error(err);
    summaryBox.textContent = "Error: Could not summarize. Please check the URL and try again.";
  } finally {
    // Reset button state
    btn.disabled = false;
    btn.textContent = "Click to Summarize";
  }
});