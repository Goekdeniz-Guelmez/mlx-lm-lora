const methods = {
  sft: {
    label: "Supervised fine-tuning",
    title: "Teach the model a sharper way to speak.",
    description: "Use prompt and completion pairs or chat messages to turn a capable base model into one that understands your task, tone, and format.",
    command: "mlx_lm_lora.train \\\n+  --model <model> \\\n+  --train-mode sft \\\n+  --data <dataset>",
    link: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#supervised-fine-tuning-sft"
  },
  dpo: {
    label: "Direct preference optimization",
    title: "Pull behavior toward the better answer.",
    description: "Train directly on chosen and rejected responses, so preference data becomes a clear, efficient signal for the model.",
    command: "mlx_lm_lora.train \\\n+  --model <model> \\\n+  --train-mode dpo \\\n+  --data <dataset>",
    link: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#direct-preference-optimization-dpo"
  },
  grpo: {
    label: "Group relative policy optimization",
    title: "Let the reward shape the next move.",
    description: "Use grouped generations and a custom reward function to optimize policy behavior without a separate critic model.",
    command: "mlx_lm_lora.train \\\n+  --model <model> \\\n+  --train-mode grpo \\\n+  --data <dataset>",
    link: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#group-relative-policy-optimization-grpo"
  },
  orpo: {
    label: "Odds ratio preference optimization",
    title: "Prefer the right answer in one pass.",
    description: "A monolithic preference objective that combines supervised learning and preference alignment without a reference model.",
    command: "mlx_lm_lora.train \\\n+  --model <model> \\\n+  --train-mode orpo \\\n+  --data <dataset>",
    link: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#odds-ratio-preference-optimization-orpo"
  },
  online: {
    label: "Online direct preference optimization",
    title: "Keep the feedback loop alive.",
    description: "Generate, judge, and update in an online preference workflow when the best data is the data you collect as you go.",
    command: "mlx_lm_lora.train \\\n+  --model <model> \\\n+  --train-mode online_dpo \\\n+  --data <dataset>",
    link: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora#online-dpo"
  }
};

const methodTabs = document.querySelectorAll(".method-tab");
const methodLabel = document.querySelector("#method-label");
const methodTitle = document.querySelector("#method-title");
const methodDescription = document.querySelector("#method-description");
const methodCommand = document.querySelector("#method-command");
const methodStatus = document.querySelector("#method-status");
const methodLink = document.querySelector("#method-link");

function selectMethod(key) {
  const method = methods[key];
  if (!method) return;

  methodTabs.forEach((tab) => {
    const isActive = tab.dataset.method === key;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  methodLabel.textContent = method.label;
  methodTitle.textContent = method.title;
  methodDescription.textContent = method.description;
  methodCommand.textContent = method.command;
  methodStatus.textContent = key === "online" ? "ONLINE DPO" : key.toUpperCase();
  methodLink.href = method.link;
}

methodTabs.forEach((tab) => {
  tab.addEventListener("click", () => selectMethod(tab.dataset.method));
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent);
      button.classList.add("is-copied");
      button.innerHTML = '<span class="copy-icon" aria-hidden="true">✓</span> Copied';
      window.setTimeout(() => {
        button.classList.remove("is-copied");
        button.innerHTML = '<span class="copy-icon" aria-hidden="true">□</span> Copy';
      }, 1800);
    } catch {
      button.textContent = "Select to copy";
    }
  });
});

const menuToggle = document.querySelector(".menu-toggle");
const mobileNav = document.querySelector(".mobile-nav");

function closeMenu() {
  menuToggle.classList.remove("is-open");
  menuToggle.setAttribute("aria-expanded", "false");
  mobileNav.classList.remove("is-open");
}

menuToggle.addEventListener("click", () => {
  const isOpen = menuToggle.classList.toggle("is-open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
  mobileNav.classList.toggle("is-open", isOpen);
});

mobileNav.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));

const revealItems = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const observer = new IntersectionObserver((entries, instance) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        instance.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}
