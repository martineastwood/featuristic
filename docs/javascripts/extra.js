// Configure MathJax before loading
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    renderActions: {
      addMenu: [0, '', '']
    }
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      // Typeset all arithmatex elements
      document.querySelectorAll('.arithmatex').forEach((el) => {
        MathJax.typesetPromise([el]);
      });
    }
  }
};

// Load MathJax
(function() {
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.async = true;
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  var firstScript = document.getElementsByTagName('script')[0];
  firstScript.parentNode.insertBefore(script, firstScript);
})();

// Custom JavaScript for Featuristic documentation
document$.subscribe(function() {
  // Custom initialization code
  console.log("Featuristic documentation loaded");
});
