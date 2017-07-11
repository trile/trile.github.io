// document.ready(function(){
  document.querySelector('#hamburger').addEventListener('click', function(){
    console.log("hello");
    var nav = document.querySelector('#main-nav');
    console.log(nav);
    nav.classList.toggle('open');
  });
// });