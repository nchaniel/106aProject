// Highlight the current section in the sticky nav.
const navLinks = Array.from(document.querySelectorAll('.nav a'));
const sections = navLinks
  .map(a => document.querySelector(a.getAttribute('href')))
  .filter(Boolean);

const setActive = (id) => {
  navLinks.forEach(a => {
    const match = a.getAttribute('href') === '#' + id;
    a.style.color = match ? 'var(--accent-deep)' : '';
    a.style.fontWeight = match ? '700' : '';
  });
};

if ('IntersectionObserver' in window && sections.length) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) setActive(e.target.id);
    });
  }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
  sections.forEach(s => io.observe(s));
}
