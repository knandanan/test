image_name=classifierv2

.PHONY: buildd
help:
	@echo 
	@echo "\033[42m= = = = = = = = = = = = = = = = Help = = = = = = = = = = = = = = = =\033[0m"
	@echo "\033[31mBoth 'dev' and 'prod' dockers gets build"
	@echo " - To build a docker: 'make buildd'"
	@echo
buildd:
	docker build --target dev -f docker/Dockerfile -t ${image_name}:dev .
	docker build --target prod -f docker/Dockerfile -t ${image_name}:prod .
