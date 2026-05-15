/* =========================================================
   Scroll-reveal — auto-tag section content and observe it
   ========================================================= */
(function initReveal() {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Auto-add data-reveal to common content blocks
  const selectors = [
    '.section-head',
    '.phase-card',
    '.kpi-row li',
    '.gallery figure',
    '.diagram-wrap',
    '.detail-panel',
    '.two-col .figure',
    '.placeholder',
    '.status-card',
    '.hero-text > *',
    '.carousel'
  ];
  const els = [];
  selectors.forEach(sel => {
    document.querySelectorAll(sel).forEach((el, i) => {
      if (!el.hasAttribute('data-reveal')) {
        el.setAttribute('data-reveal', '');
        // Stagger siblings within a group
        const idx = Array.prototype.indexOf.call(el.parentNode.children, el);
        el.setAttribute('data-delay', String(Math.min(idx, 4) * 100));
      }
      els.push(el);
    });
  });

  if (reduced) {
    els.forEach(el => el.classList.add('is-visible'));
    return;
  }
  if (!('IntersectionObserver' in window)) {
    els.forEach(el => el.classList.add('is-visible'));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('is-visible');
        io.unobserve(e.target);
      }
    });
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.12 });
  els.forEach(el => io.observe(el));
})();

/* =========================================================
   Sticky-nav active section highlight
   ========================================================= */
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

/* =========================================================
   Hero carousel
   ========================================================= */
(function initCarousels() {
  const allTracks = document.querySelectorAll('.carousel-track');
  if (!allTracks.length) return;

  allTracks.forEach(track => {
    const slides = Array.from(track.querySelectorAll('.carousel-slide'));
    const carousel = track.closest('.carousel');
    const dots = Array.from(carousel.querySelectorAll('.carousel-dot'));
    const prev = carousel.querySelector('.carousel-prev');
    const next = carousel.querySelector('.carousel-next');
    let i = 0;
    let timer = null;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const go = (n) => {
      i = (n + slides.length) % slides.length;
      slides.forEach((s, k) => s.classList.toggle('is-active', k === i));
      dots.forEach((d, k) => d.classList.toggle('is-active', k === i));
    };

    const advance = () => go(i + 1);

    const start = () => {
      if (reducedMotion) return;
      stop();
      timer = setInterval(advance, 4500);
    };
    const stop = () => { if (timer) { clearInterval(timer); timer = null; } };

    prev?.addEventListener('click', () => { go(i - 1); start(); });
    next?.addEventListener('click', () => { go(i + 1); start(); });
    dots.forEach((d) => d.addEventListener('click', () => {
      go(parseInt(d.dataset.i, 10));
      start();
    }));

    // Pause on hover/focus
    carousel?.addEventListener('mouseenter', stop);
    carousel?.addEventListener('mouseleave', start);
    carousel?.addEventListener('focusin', stop);
    carousel?.addEventListener('focusout', start);

    // Keyboard arrows while focused on carousel
    carousel?.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft')  { go(i - 1); start(); }
      if (e.key === 'ArrowRight') { go(i + 1); start(); }
    });

    start();
  });
})();

/* =========================================================
   Clickable architecture diagram → detail panel
   ========================================================= */
(function initArchDetail() {
  const panel = document.getElementById('detailPanel');
  if (!panel) return;
  const nodes = Array.from(document.querySelectorAll('.arch-node'));
  const emptyState = panel.querySelector('.detail-empty');

  const openDetail = (key) => {
    const tpl = document.getElementById('tpl-' + key);
    if (!tpl) return;
    // Clear any previous body
    panel.querySelectorAll('.detail-body').forEach(el => el.remove());
    // Clone template content into a wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'detail-body';
    wrapper.appendChild(tpl.content.cloneNode(true));
    if (emptyState) emptyState.style.display = 'none';
    panel.appendChild(wrapper);
    panel.classList.add('has-content');

    // Mark active node
    nodes.forEach(n => n.classList.toggle(
      'is-active',
      n.getAttribute('data-detail') === key
    ));

    // Smooth-scroll the panel into view
    const rect = panel.getBoundingClientRect();
    const offBottom = rect.bottom > window.innerHeight;
    const offTop = rect.top < 80;
    if (offBottom || offTop) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  nodes.forEach(n => {
    n.addEventListener('click', () => openDetail(n.getAttribute('data-detail')));
    n.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openDetail(n.getAttribute('data-detail'));
      }
    });
  });

  // In-page links like data-open-detail="armcircler"
  document.querySelectorAll('[data-open-detail]').forEach(a => {
    a.addEventListener('click', (e) => {
      const key = a.getAttribute('data-open-detail');
      // Let the href="#architecture" handle the scroll, then open detail
      setTimeout(() => openDetail(key), 350);
    });
  });
})();
