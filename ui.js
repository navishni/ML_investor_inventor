function setupPasswordToggles() {
  document.querySelectorAll(".password-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.target);
      const showing = target.type === "text";
      target.type = showing ? "password" : "text";
      button.textContent = showing ? "Show" : "Hide";
      button.setAttribute("aria-label", showing ? "Show password" : "Hide password");
    });
  });
}

function setupSignupEntitySelect() {
  const roleSelect = document.getElementById("role");
  const entitySelect = document.getElementById("entity_id");
  if (!roleSelect || !entitySelect) return;

  const investorOptions = JSON.parse(document.getElementById("investor-options").textContent);
  const inventorOptions = JSON.parse(document.getElementById("inventor-options").textContent);

  const renderOptions = () => {
    const role = roleSelect.value;
    const rows = role === "investor" ? investorOptions : inventorOptions;
    const valueKey = role === "investor" ? "investor_id" : "idea_id";
    const labelKey = role === "investor" ? "investor_name" : "idea_title";
    entitySelect.innerHTML = `<option value="">Select one</option>` + rows.map(
      (row) => `<option value="${row[valueKey]}">${row[labelKey]}</option>`
    ).join("");
  };

  roleSelect.addEventListener("change", renderOptions);
  renderOptions();
}

setupPasswordToggles();
setupSignupEntitySelect();
