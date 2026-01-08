-- public.product_categories definition

-- Drop table

-- DROP TABLE public.product_categories;

CREATE TABLE public.product_categories (
	category_id serial4 NOT NULL,
	category_name varchar NOT NULL,
	CONSTRAINT product_categories_category_name_key UNIQUE (category_name),
	CONSTRAINT product_categories_pkey PRIMARY KEY (category_id)
);


-- public.sub_categories definition

-- Drop table

-- DROP TABLE public.sub_categories;

CREATE TABLE public.sub_categories (
	sub_category_id serial4 NOT NULL,
	sub_category_name varchar NOT NULL,
	category_id int4 NOT NULL,
	CONSTRAINT sub_categories_category_id_sub_category_name_key UNIQUE (category_id, sub_category_name),
	CONSTRAINT sub_categories_pkey PRIMARY KEY (sub_category_id),
	CONSTRAINT sub_categories_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.product_categories(category_id) ON DELETE RESTRICT
);
CREATE INDEX idx_sub_categories_category_id ON public.sub_categories USING btree (category_id);


-- public.dropdown_options definition

-- Drop table

-- DROP TABLE public.dropdown_options;

CREATE TABLE public.dropdown_options (
	sub_category_id int4 NOT NULL,
	dropdown_name varchar NOT NULL,
	"option" varchar NOT NULL,
	"index" int4 NOT NULL,
	CONSTRAINT dropdown_options_pkey PRIMARY KEY (sub_category_id, dropdown_name, option),
	CONSTRAINT dropdown_options_sub_category_id_fkey FOREIGN KEY (sub_category_id) REFERENCES public.sub_categories(sub_category_id) ON DELETE CASCADE
);


-- public.forms definition

-- Drop table

-- DROP TABLE public.forms;

CREATE TABLE public.forms (
	form_id serial4 NOT NULL,
	form_name varchar NOT NULL,
	sub_category_id int4 NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT forms_form_name_sub_category_id_key UNIQUE (form_name, sub_category_id),
	CONSTRAINT forms_pkey PRIMARY KEY (form_id),
	CONSTRAINT forms_sub_category_id_fkey FOREIGN KEY (sub_category_id) REFERENCES public.sub_categories(sub_category_id) ON DELETE RESTRICT
);
CREATE INDEX idx_forms_form_name ON public.forms USING btree (form_name);
CREATE INDEX idx_forms_sub_category_id ON public.forms USING btree (sub_category_id);


-- public.spc_rules definition

-- Drop table

-- DROP TABLE public.spc_rules;

CREATE TABLE public.spc_rules (
	sub_category_id int4 NOT NULL,
	form_id int4 NOT NULL,
	rule_index int4 NOT NULL,
	rule_number_1 int4 NOT NULL,
	rule_number_2 int4 NULL,
	is_enabled bool NOT NULL,
	CONSTRAINT spc_rules_pkey PRIMARY KEY (sub_category_id, form_id, rule_index),
	CONSTRAINT spc_rules_form_id_fkey FOREIGN KEY (form_id) REFERENCES public.forms(form_id) ON DELETE CASCADE,
	CONSTRAINT spc_rules_sub_category_id_fkey FOREIGN KEY (sub_category_id) REFERENCES public.sub_categories(sub_category_id) ON DELETE CASCADE
);
CREATE INDEX idx_spc_rules_form_id ON public.spc_rules USING btree (form_id);
CREATE INDEX idx_spc_rules_rule_number_1 ON public.spc_rules USING btree (rule_number_1);
CREATE INDEX idx_spc_rules_rule_number_2 ON public.spc_rules USING btree (rule_number_2);
CREATE INDEX idx_spc_rules_sub_category_id ON public.spc_rules USING btree (sub_category_id);


-- public.work_stations definition

-- Drop table

-- DROP TABLE public.work_stations;

CREATE TABLE public.work_stations (
	station_id serial4 NOT NULL,
	form_id int4 NOT NULL,
	station_name varchar NOT NULL,
	CONSTRAINT form_work_stations_form_id_station_name_key UNIQUE (form_id, station_name),
	CONSTRAINT form_work_stations_pkey PRIMARY KEY (station_id),
	CONSTRAINT form_work_stations_form_id_fkey FOREIGN KEY (form_id) REFERENCES public.forms(form_id) ON DELETE CASCADE
);


-- public.form_fields definition

-- Drop table

-- DROP TABLE public.form_fields;

CREATE TABLE public.form_fields (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	product_characteristic varchar NOT NULL,
	graph_area varchar NULL,
	target_spec varchar NULL,
	upper_limit numeric NULL,
	lower_limit numeric NULL,
	unit varchar NULL,
	measuring_method varchar NULL,
	sample_size int4 NOT NULL,
	frequency varchar NULL,
	control_method varchar NULL,
	remark text NULL,
	CONSTRAINT form_fields_pkey PRIMARY KEY (form_id, field_index),
	CONSTRAINT form_fields_form_id_fkey FOREIGN KEY (form_id) REFERENCES public.forms(form_id) ON DELETE RESTRICT
);
CREATE INDEX idx_form_fields_form_id ON public.form_fields USING btree (form_id);


-- public.spc_measurements definition

-- Drop table

-- DROP TABLE public.spc_measurements;

CREATE TABLE public.spc_measurements (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	sample_index int4 NOT NULL,
	value numeric NOT NULL,
	work_order_count int4 NULL,
	work_order varchar NULL,
	machine_id varchar NULL,
	shift varchar NULL,
	inspector varchar NULL,
	reason varchar NULL,
	line_name varchar NOT NULL,
	inspect_time timestamptz NOT NULL,
	part_no varchar NULL,
	other_remark varchar NULL,
	CONSTRAINT spc_measurements_pkey PRIMARY KEY (form_id, field_index, sample_index, inspect_time, line_name),
	CONSTRAINT spc_measurements_form_id_field_index_fkey FOREIGN KEY (form_id,field_index) REFERENCES public.form_fields(form_id,field_index) ON DELETE RESTRICT
);
CREATE INDEX idx_spc_measurements_form_field ON public.spc_measurements USING btree (form_id, field_index);
CREATE INDEX idx_spc_measurements_inspect_time ON public.spc_measurements USING btree (inspect_time);


-- public.work_order_groups definition

-- Drop table

-- DROP TABLE public.work_order_groups;

CREATE TABLE public.work_order_groups (
	id serial4 NOT NULL,
	group_key varchar NOT NULL,
	work_order varchar NOT NULL,
	work_order_count int4 NULL,
	form_id int4 NULL,
	field_index int4 NULL,
	sample_index int4 NULL,
	inspect_time timestamptz NULL,
	line_name varchar NULL,
	CONSTRAINT uq_group UNIQUE (group_key, work_order),
	CONSTRAINT work_order_groups_pkey PRIMARY KEY (id),
	CONSTRAINT fk_work_order_groups_measurement FOREIGN KEY (form_id,field_index,sample_index,inspect_time,line_name) REFERENCES public.spc_measurements(form_id,field_index,sample_index,inspect_time,line_name) ON DELETE CASCADE
);
CREATE INDEX idx_work_order_groups_group_key ON public.work_order_groups USING btree (group_key);
CREATE INDEX idx_work_order_groups_work_order ON public.work_order_groups USING btree (work_order);


-- public.alert_logs definition

-- Drop table

-- DROP TABLE public.alert_logs;

CREATE TABLE public.alert_logs (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	sample_index int4 NOT NULL,
	inspect_time timestamptz NOT NULL,
	line_name varchar NOT NULL,
	alert_type varchar NOT NULL,
	reason text NULL,
	reason_recorded_at timestamptz NULL,
	CONSTRAINT alert_logs_pkey PRIMARY KEY (form_id, field_index, sample_index, inspect_time, line_name, alert_type),
	CONSTRAINT fk_alert_logs_measurement FOREIGN KEY (form_id,field_index,sample_index,inspect_time,line_name) REFERENCES public.spc_measurements(form_id,field_index,sample_index,inspect_time,line_name) ON DELETE CASCADE
);
CREATE INDEX idx_alert_logs_form ON public.alert_logs USING btree (form_id, field_index);
CREATE INDEX idx_alert_logs_time ON public.alert_logs USING btree (inspect_time);


-- public.control_limits definition

-- Drop table

-- DROP TABLE public.control_limits;

CREATE TABLE public.control_limits (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	calculate_method varchar NOT NULL,
	upper_control_limit numeric NULL,
	lower_control_limit numeric NULL,
	CONSTRAINT control_limits_calculate_method_check CHECK (((calculate_method)::text = ANY (ARRAY[('control_limit_formula'::character varying)::text, ('std_dev'::character varying)::text, ('self_define_limit'::character varying)::text]))),
	CONSTRAINT control_limits_pkey PRIMARY KEY (form_id, field_index),
	CONSTRAINT fk_control_limits_form_field FOREIGN KEY (form_id,field_index) REFERENCES public.form_fields(form_id,field_index) ON DELETE CASCADE
);


-- public.field_formulas definition

-- Drop table

-- DROP TABLE public.field_formulas;

CREATE TABLE public.field_formulas (
	formula_id serial4 NOT NULL,
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	formula text NOT NULL,
	block_number int4 NOT NULL,
	save_all_calculate_value bool DEFAULT false NULL,
	CONSTRAINT field_formulas_pkey PRIMARY KEY (formula_id),
	CONSTRAINT fk_form_field FOREIGN KEY (form_id,field_index) REFERENCES public.form_fields(form_id,field_index) ON DELETE CASCADE
);


-- public.field_work_stations definition

-- Drop table

-- DROP TABLE public.field_work_stations;

CREATE TABLE public.field_work_stations (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	station_id int4 NOT NULL,
	CONSTRAINT field_work_stations_pkey PRIMARY KEY (form_id, field_index, station_id),
	CONSTRAINT field_work_stations_form_id_field_index_fkey FOREIGN KEY (form_id,field_index) REFERENCES public.form_fields(form_id,field_index) ON DELETE CASCADE,
	CONSTRAINT field_work_stations_station_id_fkey FOREIGN KEY (station_id) REFERENCES public.work_stations(station_id) ON DELETE CASCADE
);


-- public.calculated_formula_values definition

-- Drop table

-- DROP TABLE public.calculated_formula_values;

CREATE TABLE public.calculated_formula_values (
	form_id int4 NOT NULL,
	field_index int4 NOT NULL,
	sample_index int4 NOT NULL,
	inspect_time timestamptz NOT NULL,
	line_name varchar NOT NULL,
	formula_id int4 NULL,
	"key" text NOT NULL,
	value numeric NOT NULL,
	CONSTRAINT calculated_formula_values_pkey PRIMARY KEY (form_id, field_index, sample_index, inspect_time, line_name, key),
	CONSTRAINT calculated_formula_values_form_id_field_index_sample_index_fkey FOREIGN KEY (form_id,field_index,sample_index,inspect_time,line_name) REFERENCES public.spc_measurements(form_id,field_index,sample_index,inspect_time,line_name) ON DELETE CASCADE,
	CONSTRAINT calculated_formula_values_formula_id_fkey FOREIGN KEY (formula_id) REFERENCES public.field_formulas(formula_id) ON DELETE SET NULL
);