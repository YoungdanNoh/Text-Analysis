package com.dan.recture.persistance;

import java.util.List;

import com.dan.recture.domain.AirbnbResultVO;


public interface AirbnbResultDAO {
	List<AirbnbResultVO> selectResult() throws Exception;
}
