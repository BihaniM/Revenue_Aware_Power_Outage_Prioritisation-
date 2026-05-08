document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chatForm');
  const answer = document.getElementById('chatAnswer');
  if (!form) return;
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    answer.textContent = 'Analyzing database priority and available resources...';
    const body = new FormData(form);
    try {
      const response = await fetch('/api/chat', { method: 'POST', body });
      const data = await response.json();
      answer.textContent = data.answer || 'No answer generated.';
    } catch (err) {
      answer.textContent = 'Chat service error: ' + err;
    }
  });
});
