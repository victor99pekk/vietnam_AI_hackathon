/* ============================================================
   DataForge Pitch Deck — Navigation
   ============================================================ */

(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const total = slides.length;
  let current = 0;
  let transitioning = false;

  const counter = document.getElementById('slideCounter');

  function updateCounter() {
    counter.textContent = total > 1 ? `${current + 1} / ${total}` : `${current + 1}`;
  }

  function goTo(index) {
    if (transitioning || index === current || index < 0 || index >= total) return;
    transitioning = true;

    const oldSlide = slides[current];
    const newSlide = slides[index];

    oldSlide.classList.remove('active');
    newSlide.classList.add('active');

    current = index;
    updateCounter();

    setTimeout(() => { transitioning = false; }, 450);
  }

  function next() { goTo(current + 1); }
  function prev() { goTo(current - 1); }

  // Keyboard
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
      e.preventDefault();
      next();
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      prev();
    }
  });

  // Mouse wheel
  let wheelTimeout;
  document.addEventListener('wheel', (e) => {
    if (wheelTimeout) return;
    wheelTimeout = setTimeout(() => { wheelTimeout = null; }, 700);
    if (e.deltaY > 20) next();
    else if (e.deltaY < -20) prev();
  }, { passive: true });

  updateCounter();
  console.log('%c📊 Vietnamese KG Pitch %c| %c← → or scroll',
    'color: #2563eb; font-weight: bold;', '', 'color: #888;');
})();
