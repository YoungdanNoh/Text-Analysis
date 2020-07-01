<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib prefix="c"   uri="http://java.sun.com/jsp/jstl/core" %>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>뉴스 텍스트 분석 결과 차트</title>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      var category = new Array();
	  var all_data = new Array();
	  var correct = new Array();
	
	  <c:forEach items="${list }" var="TextAnalysisModel">
		  category.push("${TextAnalysisModel.category }");
		  all_data.push("${TextAnalysisModel.all_data }");
		  correct.push("${TextAnalysisModel.correct_data }");
	  </c:forEach>
      
      google.charts.load('current', {'packages':['bar']});
      google.charts.setOnLoadCallback(drawChart);

  	  var a= true;
      function drawChart() {

        var data = google.visualization.arrayToDataTable([
          ['Category', 'ALL_DATA', 'Correct'],
          [category[0], Number(all_data[0]), Number(correct[0])],
          [category[1], Number(all_data[1]), Number(correct[1])],
          [category[2], Number(all_data[2]), Number(correct[2])],
          
        ]);

        var options = {
          chart: {
            title: 'Result',
            subtitle: 'Compare All_Data with Correct_Data',
          }
        };

        var chart = new google.charts.Bar(document.getElementById('columnchart_material'));

        chart.draw(data, google.charts.Bar.convertOptions(options));
      }
    </script>
</head>
<body>
	<h1>훈련 결과</h1>
    <br>
	<c:forEach items="${list }" begin="1" end="1" var="TextAnalysisModel">
		<h2>모델 정확도 : ${TextAnalysisModel.evaluate*100 }%</h2>
	</c:forEach>
	<div id="columnchart_material" style="width: 800px; height: 500px;"></div>
	
	<c:set var="all_data" value="0" />
	<c:set var="correct_data" value="0" />
	<c:forEach items="${list }" var="TextAnalysisModel">
		  <h2>${TextAnalysisModel.category } 카테고리 정확도: 
		  <c:set var="accuracy" value="${100/TextAnalysisModel.all_data*TextAnalysisModel.correct_data}"/>
		  <c:out value="${accuracy}"/>%</h2>
		  <c:set var="all_data" value="${all_data + TextAnalysisModel.all_data}" />
		  <c:set var="correct_data" value="${correct_data + TextAnalysisModel.correct_data}" />
	</c:forEach>
	<h3>전체 데이터 개수: <c:out value="${all_data }"/>개</h3>
	<h3>맞은 데이터 개수: <c:out value="${correct_data }"/>개</h3>
</body>
</html>