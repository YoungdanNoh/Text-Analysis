package com.dan.recture.domain;

public class AirbnbResultVO {
	/* *
	 * id
	 * host_id
	 * accommodates
	 * bathrooms
	 * bedrooms
	 * beds
	 * predicted_price
	 * */
	
	private double id;
	private double host_id;
	private double accommodates;
	private double bathrooms;
	private double bedrooms;
	private double beds;
	private double predicted_price;
	
	public double getId() {
		return id;
	}
	public void setId(double id) {
		this.id = id;
	}
	public double getHost_id() {
		return host_id;
	}
	public void setHost_id(double host_id) {
		this.host_id = host_id;
	}
	public double getAccommodates() {
		return accommodates;
	}
	public void setAccommodates(double accommodates) {
		this.accommodates = accommodates;
	}
	public double getBathrooms() {
		return bathrooms;
	}
	public void setBathrooms(double bathrooms) {
		this.bathrooms = bathrooms;
	}
	public double getBedrooms() {
		return bedrooms;
	}
	public void setBedrooms(double bedrooms) {
		this.bedrooms = bedrooms;
	}
	public double getBeds() {
		return beds;
	}
	public void setBeds(double beds) {
		this.beds = beds;
	}
	public double getPredicted_price() {
		return predicted_price;
	}
	public void setPredicted_price(double predicted_price) {
		this.predicted_price = predicted_price;
	}
	
}
