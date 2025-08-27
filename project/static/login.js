// role toggle
const roleToggle = document.getElementById('roleToggle');
const roleButtons = [...document.querySelectorAll('.role-btn')];
const emailLabel = document.getElementById('emailLabel');
const emailInput = document.getElementById('emailInput');
const signinBtn = document.getElementById('signinBtn').querySelector('span');

let currentRole = 'admin';

roleToggle.addEventListener('click', (e) => {
  const btn = e.target.closest('.role-btn');
  if (!btn) return;
  roleButtons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentRole = btn.dataset.role;

  if (currentRole === 'admin') {
    emailLabel.textContent = 'Administrator Email';
    emailInput.placeholder = 'admin@baymax.com';
    signinBtn.textContent = 'Sign In as Administrator';
  } else {
    emailLabel.textContent = 'Hospital Email';
    emailInput.placeholder = 'hospital@baymax.com';
    signinBtn.textContent = 'Sign In as Hospital';
  }
});

// no password check – simple redirect based on role
document.getElementById('loginForm').addEventListener('submit', (e) => {
  e.preventDefault();
  window.location.href = currentRole === 'admin' ? 'admin.html' : 'hospital.html';
});
