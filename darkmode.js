/* Dark mode toggle, mobile burger menu, and auth-aware nav links */
(function() {
  // Apply saved preference immediately (before paint)
  const saved = localStorage.getItem('oncovision-theme');
  if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');

  function injectToggle() {
    const nav = document.querySelector('.nav__inner');
    if (!nav) return;

    // Don't inject twice
    if (document.getElementById('darkToggleBtn')) return;

    const btn = document.createElement('button');
    btn.id = 'darkToggleBtn';
    btn.className = 'dark-toggle';
    btn.setAttribute('aria-label', 'Toggle dark mode');
    btn.innerHTML = `<span class="dark-toggle__icon" id="darkIcon">🌙</span><span id="darkLabel">Dark</span>`;

    btn.addEventListener('click', () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('oncovision-theme', 'light');
        document.getElementById('darkLabel').textContent = 'Dark';
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('oncovision-theme', 'dark');
        document.getElementById('darkLabel').textContent = 'Light';
      }
      updateIcon();
    });

    // Insert before the CTA link (last li)
    const links = nav.querySelector('.nav__links');
    if (links) {
      const li = document.createElement('li');
      li.appendChild(btn);
      links.appendChild(li);
    } else {
      nav.appendChild(btn);
    }

    updateIcon();
  }

  function updateIcon() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const icon = document.getElementById('darkIcon');
    const label = document.getElementById('darkLabel');
    if (icon) icon.textContent = isDark ? '☀️' : '🌙';
    if (label) label.textContent = isDark ? 'Light' : 'Dark';
  }

  function injectBurger() {
    const nav = document.querySelector('.nav');
    const navInner = nav ? nav.querySelector('.nav__inner') : null;
    if (!navInner) return;
    if (document.getElementById('navBurgerBtn')) return;

    const burger = document.createElement('button');
    burger.id = 'navBurgerBtn';
    burger.className = 'nav__burger';
    burger.setAttribute('aria-label', 'Toggle navigation');
    burger.setAttribute('aria-expanded', 'false');
    burger.innerHTML = '<span></span><span></span><span></span>';

    burger.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('nav--open');
      burger.setAttribute('aria-expanded', String(isOpen));
    });

    navInner.appendChild(burger);

    // Close menu when any nav link is clicked
    navInner.querySelectorAll('.nav__links a').forEach(link => {
      link.addEventListener('click', () => {
        nav.classList.remove('nav--open');
        burger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  function updateNav() {
    // Hide protected links immediately before auth check to prevent flash
    document.querySelectorAll('[data-auth="true"]').forEach(el => el.style.display = 'none');

    fetch('/api/check-auth', {credentials: 'include'})
      .then(r => r.json())
      .then(d => {
        document.querySelectorAll('[data-auth="true"]').forEach(el => {
          el.style.display = d.logged_in ? '' : 'none';
        });
      })
      .catch(() => {
        document.querySelectorAll('[data-auth="true"]').forEach(el => el.style.display = 'none');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { injectToggle(); injectBurger(); updateNav(); });
  } else {
    injectToggle();
    injectBurger();
    updateNav();
  }
})();