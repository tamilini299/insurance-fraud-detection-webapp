// logout
document.getElementById('logout-btn-hospital')?.addEventListener('click', (e)=>{
  e.preventDefault();
  window.location.href = 'index.html';
});

// toast helper
const toast = document.getElementById('toast');
function showToast(msg){
  document.getElementById('toastText').textContent = msg || 'Done';
  toast.classList.remove('hidden');
  setTimeout(()=> toast.classList.add('hidden'), 2000);
}

// submit (simple validation; ready to wire to Azure later)
document.getElementById('submitClaim')?.addEventListener('click', ()=>{
  const providerId = document.getElementById('providerIdClaim').value.trim();
  const insAmount = parseFloat(document.getElementById('insAmount').value);
  const dedAmount = parseFloat(document.getElementById('dedAmount').value);
  const details = document.getElementById('claimDetails').value.trim();
  const attending = document.getElementById('attendingPhysician').value.trim();
  const operating = document.getElementById('operatingPhysician').value.trim();
  const other = document.getElementById('otherPhysician').value.trim();

  // collect demographics
  const gender = document.querySelector('input[name="gender"]:checked')?.value || null;
  const race = document.querySelector('input[name="race"]:checked')?.value || null;
  const renal = document.getElementById('renal').checked ? 1 : 0;

  // collect chronic conditions
  const conditions = [...document.querySelectorAll('input[name="condition"]:checked')]
                        .map(el => el.parentElement.textContent.trim());

  // collect procedure codes
const procedures = [...document.querySelectorAll('#procSection .input-field')]
                      .map(el => el.value.trim()).filter(v => v);

// collect diagnosis codes
const diagnoses = [...document.querySelectorAll('#diagSection .input-field')]
                      .map(el => el.value.trim()).filter(v => v);


  const payload = {
    providerId,
    insAmount,
    dedAmount,
    details,
    attendingPhysician: attending,
    operatingPhysician: operating,
    otherPhysician: other,
    gender,
    race,
    renal,
    conditions,
    procCodes: procedures,
    diagCodes: diagnoses
  };

  console.log("Submitting claim:", payload);

  fetch("http://127.0.0.1:8000/submit-claim", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  .then(res => res.json())
  .then(data => {
    console.log("Server response:", data);
    showToast(data.message || "Claim submitted");
  })
  .catch(err => {
    console.error("Error submitting claim:", err);
    showToast("Error submitting claim");
  });
});

