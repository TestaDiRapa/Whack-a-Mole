console.log("loaded!");
const form = document.forms.namedItem("settings")

form.addEventListener('submit', function(ev){

  outDiv = document.querySelector("div#result");
  outDiv.innerHTML = "";

  loadDiv = document.querySelector("#loading");
  loadDiv.style.display = 'block';

  errDiv = document.querySelector("#error");
  errDiv.style.display = 'none';

  fData = new FormData(form);

  var xmlHttr = new XMLHttpRequest();
  xmlHttr.open("POST", "/process", true)

  xmlHttr.onload = function(e) {
    loadDiv.style.display = 'none';
    if(xmlHttr.status == 200) {
      outDiv.innerHTML = xmlHttr.response;
    } else {
      errDiv.style.display = 'block';
    }
  }

  xmlHttr.onerror = function(e) {
    loadDiv.style.display = 'none';
    errDiv.style.display = 'block';
  }

  xmlHttr.send(fData);

  ev.preventDefault();

}, false);