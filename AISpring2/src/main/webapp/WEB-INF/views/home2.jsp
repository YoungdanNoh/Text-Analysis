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
        <div class="row align-items-center text-center justify-content-center">
          <div class="col-lg-8 col-md-8">
           <!--  <a data-fancybox data-ratio="2" href="https://vimeo.com/317571768" class="play-button d-block">
              <span class="icon-play"></span>
            </a> -->
            <br><br><br><br>

				 <h1 class="text" style="color: #0F4C81">Airbnb 가격 산정</h1>
				<form action="fileupload2" method="post" enctype="multipart/form-data">
					<br>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">* 업로드 파일 내용 *</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">1. 형식- csv</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">2. 포함할 열(6개)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">id(숙소 id)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">host_id(사용자 id)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">accommodates(수용인원)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">bathrooms(화장실 수)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">bedrooms(침실 수)</span>
				    <span class="sub-text mb-1 d-block" style="color:#0F4C81">beds(침대 개수)</span>
				    <span class="sub-text mb-3 d-block" style="color:#0F4C81">3. 각 열의 항목의 값은 숫자.</span>
				    <hr>
				    <input class="btn btn-info btn-xs" type="file" name="uploadfile" placeholder="파일 선택" /><br/>
          
				    <br>
				    <input class="btn btn-info btn-md" type="submit" value="업로드">
				</form>
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