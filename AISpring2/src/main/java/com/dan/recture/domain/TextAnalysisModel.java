package com.dan.recture.domain;

import java.util.Date;

public class TextAnalysisModel {
	
//	private double evaluate;
//	private int correct;
//	private int all_data;
//	public double getEvaluate() {
//		return evaluate;
//	}
//	public void setEvaluate(double evaluate) {
//		this.evaluate = evaluate;
//	}
//	public int getCorrect() {
//		return correct;
//	}
//	public void setCorrect(int correct) {
//		this.correct = correct;
//	}
//	public int getAll_data() {
//		return all_data;
//	}
//	public void setAll_data(int all_data) {
//		this.all_data = all_data;
//	}
	private double evaluate;
	private String Category;
	private int all_data;
	private int correct_data;
	
	public double getEvaluate() {
		return evaluate;
	}
	public void setEvaluate(double evaluate) {
		this.evaluate = evaluate;
	}
	public String getCategory() {
		return Category;
	}
	public void setCategory(String category) {
		Category = category;
	}
	public int getAll_data() {
		return all_data;
	}
	public void setAll_data(int all_data) {
		this.all_data = all_data;
	}
	public int getCorrect_data() {
		return correct_data;
	}
	public void setCorrect_data(int correct_data) {
		this.correct_data = correct_data;
	}
	

}
