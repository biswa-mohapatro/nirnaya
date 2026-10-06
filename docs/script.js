(() => {
  const root = document.documentElement;
  const themeBtn = document.querySelector('[data-theme-toggle]');
  const saved = localStorage.getItem('nirnaya-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  function setTheme(theme) {
    root.dataset.theme = theme;
    localStorage.setItem('nirnaya-theme', theme);
    if (themeBtn) {
      themeBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
      themeBtn.textContent = theme === 'dark' ? '☀︎' : '◐';
    }
  }

  setTheme(saved || (prefersDark ? 'dark' : 'light'));
  themeBtn?.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));

  const progress = document.querySelector('.progress > div');
  function updateProgress() {
    const scrollTop = window.scrollY;
    const height = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${Math.min(100, Math.max(0, (scrollTop / height) * 100))}%`;
  }
  window.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  const links = [...document.querySelectorAll('.side-nav a')];
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const observer = new IntersectionObserver(entries => {
    const visible = entries
      .filter(e => e.isIntersecting)
      .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!visible) return;
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${visible.target.id}`));
  }, { rootMargin: '-18% 0px -68% 0px', threshold: 0.01 });
  sections.forEach(section => observer.observe(section));

  document.querySelectorAll('.mobile-toc a').forEach(a => {
    a.addEventListener('click', () => {
      const details = a.closest('details');
      if (details) details.open = false;
    });
  });
})();
