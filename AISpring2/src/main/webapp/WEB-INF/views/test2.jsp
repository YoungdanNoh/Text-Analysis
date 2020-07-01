<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
<%@ page session="false"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>    

<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Archs &mdash; Onepage Template by Colorlib</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <link rel="stylesheet" href="<c:url value='https://fonts.googleapis.com/css?family=Nunito+Sans:200,300,400,700,900'/>"> 
    <link rel="stylesheet" href="resources/fonts/icomoon/style.css">
    <link rel="stylesheet" href="resources/css/bootstrap.min.css">
    <link rel="stylesheet" href="resources/css/magnific-popup.css">
    <link rel="stylesheet" href="resources/css/jquery-ui.css">
    <link rel="stylesheet" href="resources/css/owl.carousel.min.css">
    <link rel="stylesheet" href="resources/css/owl.theme.default.min.css">
    <link rel="stylesheet" href="resources/css/bootstrap-datepicker.css">
    
    <link rel="stylesheet" href="resources/fonts/flaticon/font/flaticon.css">
  
    <link rel="stylesheet" href="resources/css/aos.css">
    <link rel="stylesheet" href="resources/css/jquery.fancybox.min.css">
    
    <link rel="stylesheet" href="resources/css/style.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
 
    <style type="text/css">		
		div{
			text-align: center;

		}		
		a:link { color: black; text-decoration: none;text-align: center;}
		a:visited { color: black; text-decoration: none;text-align: center;}
		a:hover { color: #595959; text-decoration: none;text-align: center;}	
		
	</style>
	<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
	
	<c:set var="resultSum" value="0"/>    
	<c:set var="i" value="0"/>
	
    <c:forEach items="${resultList}" var="result" varStatus="st">
    	<c:set var="resultSum" value="${resultSum+result.predicted_price}"/>
    	<c:set var="i" value="${i+1}"/>
    </c:forEach>
 
    <c:set var="sumAverage" value="${resultSum/i}"/>
	
  	<script type="text/javascript">
    	google.charts.load("current", {packages:['corechart']});
    	google.charts.setOnLoadCallback(drawChart);
    	function drawChart() {
      	var data = google.visualization.arrayToDataTable([
      		["id", "예측가격", { role: "style" } ],
	        <c:forEach items="${resultList}" var="result" varStatus="st">
          	['${result.id}', parseInt('${result.predicted_price}'),"#99d6ff"],
          	</c:forEach>
      	]);

      	var view = new google.visualization.DataView(data);
     	view.setColumns([0, 1,
        	               { calc: "stringify",
            	             sourceColumn: 1,
                	         type: "string",
                    	     role: "annotation" },
                       	2]);

      	var options = {
        	title: "가격예측",
        	width: 1300,
        	height: 400,
        	bar: {groupWidth: "75%"},
        	legend: { position: "none" },
      	};
      	var chart = new google.visualization.ColumnChart(document.getElementById("columnchart_values"));
      	chart.draw(view, options);
  		}
  	</script>
	<script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawVisualization);

      function drawVisualization() {
        // Some raw data (not necessarily accurate)
        var data = google.visualization.arrayToDataTable([
        	["숙소 id", "예측가격(단위:$)", "평균가격(단위:$)" ],
	        <c:forEach items="${resultList}" var="result" varStatus="st">
          	['${result.id}', parseInt('${result.predicted_price}'),parseInt('${sumAverage}')],
          	</c:forEach>
        ]);

        var options = {
          title : 'Airbnb 가격 예측',
          vAxis: {title: '예측가격(단위:$)'},
          hAxis: {title: '숙소 id'},
          seriesType: 'bars',
          series: {1: {type: 'line'}}        };

        var chart = new google.visualization.ComboChart(document.getElementById('chart_div'));
        chart.draw(data, options);
      }
    </script>
	
</head>
  </head>
  <body data-spy="scroll" data-target=".site-navbar-target" data-offset="300">
  
  <div id="overlayer"></div>
  <div class="loader">
    <div class="spinner-border text-primary" role="status">
      <span class="sr-only">Loading...</span>
    </div>
  </div>

  <div class="site-wrap">

    <div class="site-mobile-menu site-navbar-target">
      <div class="site-mobile-menu-header">
        <div class="site-mobile-menu-close mt-3">
          <span class="icon-close2 js-menu-toggle"></span>
        </div>
      </div>
      <div class="site-mobile-menu-body"></div>
    </div> <!-- .site-mobile-menu -->
    
    
<!--     <div class="site-navbar-wrap">
      <div class="site-navbar-top">
        <div class="container py-3">
          <div class="row align-items-center">
            <div class="col-6">
              <a href="#" class="p-2 pl-0"><span class="icon-twitter"></span></a>
              <a href="#" class="p-2 pl-0"><span class="icon-facebook"></span></a>
              <a href="#" class="p-2 pl-0"><span class="icon-linkedin"></span></a>
              <a href="#" class="p-2 pl-0"><span class="icon-instagram"></span></a>
            </div>
            <div class="col-6">
              <div class="d-flex ml-auto">
                <a href="#" class="d-flex align-items-center ml-auto mr-4">
                  <span class="icon-envelope mr-2"></span>
                  <span class="d-none d-md-inline-block">info@domain.com</span>
                </a>
                <a href="#" class="d-flex align-items-center">
                  <span class="icon-phone mr-2"></span>
                  <span class="d-none d-md-inline-block">+1 234 4567 8910</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
 -->
      <div class="site-navbar-wrap site-navbar site-navbar-target js-sticky-header" style="background-color:#0F4C81;">
        <div class="site-navbar-top">
        <div class="container">
          <div class="row align-items-center">
            <div class="col-2">
              <h1 class="my-0 site-logo"><a href="index.html">Archs</a></h1>
            </div>
            <div class="col-10">
              <nav class="site-navigation text-right" role="navigation">
                <div class="container">
                  <div class="d-inline-block d-lg-none ml-md-0 mr-auto py-3">
                  	<a href="#" class="site-menu-toggle js-menu-toggle text-white"><span class="icon-menu h3"></span></a>
                  </div>

                  <ul class="site-menu main-menu js-clone-nav d-none d-lg-block">
                    <li>
                      <a href="<c:url value='/' />" class="nav-link">Home</a>
                    </li>
                    <li class="has-children">
                      <a href="#about-section" class="nav-link">About Us</a>
                      <ul class="dropdown arrow-top">
                        <li><a href="#our-team-section" class="nav-link">송우현</a></li>
                        <li><a href="#pricing-section" class="nav-link">노영단</a></li>
                        <li><a href="#faq-section" class="nav-link">노현진</a></li>              
                        <li class="has-children">
                          <a href="#">2016</a>
                          <ul class="dropdown">
                            <li><a href="#">송우현</a></li>
                            <li><a href="#">Menu Two</a></li>
                            <li><a href="#">Menu Three</a></li>
                          </ul>
                        </li>
                        <li class="has-children">
                          <a href="#">2018</a>
                          <ul class="dropdown">
                            <li><a href="#">노현진</a></li>
                            <li><a href="#">Menu Two</a></li>
                            <li><a href="#">Menu Three</a></li>
                          </ul>
                        </li>
                      </ul>
                    </li>
                    <li>
                      <a href="<c:url value='/' />" class="nav-link">Projects</a>
                    </li>
<!--                     <li><a href="#news-section" class="nav-link">News</a></li>
                    <li><a href="#services-section" class="nav-link">Services</a></li>
                    <li><a href="#contact-section" class="nav-link">Contact</a></li> -->
                  </ul>
                </div>
              </nav>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
    
    <div class="site-blocks-cover" data-stellar-background-ratio="0.5"id="home-section">
      <div class="container">
        <div class="row align-items-center text-center justify-content-center ">
          <div class="col-lg-8 col-md-8">
           <!--  <a data-fancybox data-ratio="2" href="https://vimeo.com/317571768" class="play-button d-block">
              <span class="icon-play"></span>
            </a> -->
            <br><br><br><br>
				 <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
				 <h1 class="text" style="color: #0F4C81">RESULT</h1>
				    <div id="chart_div" style="width: 65em; height: 30em;"></div>
				
			<br><br><br><br><br><br><br><br><br><br>
            <!-- <span class="sub-text mb-3 d-block" style="color:#0F4C81"><em>Lorem ipsum dolor sit amet, consectetur adipisicing elit. Quisquam molestiae ipsam, atque.</em></span>
           -->
           </div>
        </div>
        
      </div>
    </div>  
    

  
    <footer class="site-footer border-top">
      <div class="container">
        <div class="row pt-5 mt-5 text-center">
          <div class="col-md-12">
            <p>
            <!-- Link back to Colorlib can't be removed. Template is licensed under CC BY 3.0. -->
            Copyright &copy;<script>document.write(new Date().getFullYear());</script> All rights reserved | This template is made with <i class="icon-heart text-danger" aria-hidden="true"></i> by <a href="https://colorlib.com" target="_blank" >Colorlib</a>
            <!-- Link back to Colorlib can't be removed. Template is licensed under CC BY 3.0. -->
            </p>
          </div>
          
        </div>
      </div>
    </footer>

  <script src="resources/js/jquery-3.3.1.min.js"></script>
  <script src="resources/js/jquery-ui.js"></script>
  <script src="resources/js/popper.min.js"></script>
  <script src="resources/js/bootstrap.min.js"></script>
  <script src="resources/js/owl.carousel.min.js"></script>
  <script src="resources/js/jquery.countdown.min.js"></script>
  <script src="resources/js/jquery.magnific-popup.min.js"></script>
  <script src="resources/js/bootstrap-datepicker.min.js"></script>
  <script src="resources/js/aos.js"></script>
  <script src="resources/js/jquery.sticky.js"></script>
  <script src="resources/js/jquery.easing.1.3.js"></script>
  
  <script src="resources/js/jquery.fancybox.min.js"></script>
  <script src="resources/js/main.js"></script>

  
  </body>
</html>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 