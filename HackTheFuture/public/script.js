document.getElementById("year").textContent = new Date().getFullYear();

document.querySelectorAll(".dropdown .dropbtn").forEach(btn => {
  btn.addEventListener("click", e => {
    e.preventDefault();
    const dd = btn.closest(".dropdown");
    const open = dd.classList.toggle("open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  });
});
document.addEventListener("click", e => {
  document.querySelectorAll(".dropdown.open").forEach(dd => {
    if (!dd.contains(e.target)) {
      dd.classList.remove("open");
      const btn = dd.querySelector(".dropbtn");
      if (btn) btn.setAttribute("aria-expanded", "false");
    }
  });
});