"use strict";

(() => {
  const linkSelector = "[data-workspace-link]";
  const localHost = location.hostname === "127.0.0.1" || location.hostname === "localhost";
  const configured = globalThis.AUDIO_JLENS_WORKSPACES || {};

  function localUrl(port, path = "/") {
    return `${location.protocol}//${location.hostname}:${port}${path}`;
  }

  function workspaceUrls(mode) {
    const defaults = localHost
      ? {
          asr: localUrl(8000),
          speech: localUrl(8001),
          tts: localUrl(8002, "/chatterbox"),
          showcase: localUrl(8000, "/showcase"),
        }
      : {
          asr: "./",
          speech: mode === "speech" ? "./" : null,
          tts: "./chatterbox",
          showcase: "./showcase.html",
        };
    const hasShowcaseOverride = Object.prototype.hasOwnProperty.call(configured, "showcase");
    const hasLegacyCausalOverride = Object.prototype.hasOwnProperty.call(configured, "causal");
    const showcaseOverride = hasShowcaseOverride
      ? configured.showcase
      : hasLegacyCausalOverride
        ? configured.causal
        : defaults.showcase;
    return { ...defaults, ...configured, showcase: showcaseOverride };
  }

  function renderWorkspaceNavigation() {
    const mode = document.body.dataset.workspace || "asr";
    const urls = workspaceUrls(mode);
    document.querySelectorAll(linkSelector).forEach((link) => {
      const destination = link.dataset.workspaceLink;
      const url = urls[destination];
      link.hidden = !url;
      if (url) link.href = url;
      const active = destination === mode;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }

  renderWorkspaceNavigation();
  new MutationObserver(renderWorkspaceNavigation).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-workspace"],
  });
})();
