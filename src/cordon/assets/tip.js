(function () {
  var tip = document.getElementById("tip");
  if (!tip) return;
  function place(event) {
    var x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = event.clientY + 16;
    if (y + tip.offsetHeight > window.innerHeight) y = event.clientY - tip.offsetHeight - 12;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = y + "px";
  }
  document.addEventListener("pointermove", function (event) {
    var mark = event.target.closest ? event.target.closest("[data-tip]") : null;
    if (!mark || !mark.getAttribute("data-tip")) {
      tip.style.display = "none";
      return;
    }
    tip.textContent = mark.getAttribute("data-tip");
    tip.style.display = "block";
    place(event);
  });
  document.addEventListener("pointerleave", function () {
    tip.style.display = "none";
  });
})();
