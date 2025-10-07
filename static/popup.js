document.getElementById("summarizeBtn").addEventListener("click", async () => {
  const url = document.getElementById("urlInput").value;
  if (!url) {
    alert("Please paste a URL first!");
    return;
  }

  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    const data = await response.json();
    document.getElementById("summaryBox").innerText = data.summary || "No summary generated.";
  } catch (err) {
    console.error(err);
    document.getElementById("summaryBox").innerText = "Error: Could not summarize.";
  }
});
